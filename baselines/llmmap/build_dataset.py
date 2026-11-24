#!/usr/bin/env python3
"""Generate LLMmap traces (8 queries x N prompt configurations) for every model via Ollama.

Output: one JSONL per model in --out, same schema as LLMmap's make_dataset.py
({'dataset': 'train'|'test', 'llm': label, 'traces': [(query, answer)...], 'prompt_conf': {...}}).
Resumable: re-running skips configurations already written. The SAME configurations are used for every
model (seeded), so models are compared on identical prompts.

NOTE (upstream bug): PromptConfFactory.sample(n, pool) ignores `pool` and always samples from the TRAIN
pool, so LLMmap's own 'test' configs are drawn from train. We call sample_one(pool) directly so the
test configurations really come from the held-out pool.
"""
import argparse, json, os, random, sys, tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
from ollama_llm import LLM_ollama, HOST


def sample_confs(pc, n, pool, seed):  # pool constants passed in by main
    random.seed(seed)
    return [pc.sample_one(pool) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmmap", default=os.environ.get("LLMMAP_HOME", "./LLMmap"))
    ap.add_argument("--models", default="models.json")
    ap.add_argument("--only", nargs="*", help="labels to run (default: all)")
    ap.add_argument("--out", default="traces")
    ap.add_argument("--n_train", type=int, default=100, help="configs per model used for its template (LLMmap default 100)")
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel prompt configs (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--think", default="off", choices=["off", "raw"])
    ap.add_argument("--check", action="store_true", help="only verify that every tag exists in Ollama")
    a = ap.parse_args()

    sys.path.insert(0, a.llmmap)
    from LLMmap.prompt_configuration import PromptConfFactory, TRAIN, TEST   # light: no torch import
    models = json.load(open(a.models))["models"]
    have = {m["name"] for m in requests.get(f"{HOST}/api/tags", timeout=30).json()["models"]}
    have |= {n[:-7] for n in have if n.endswith(":latest")}
    missing = [m["tag"] for m in models if m["tag"] not in have]
    if a.check:
        print("missing in Ollama:", missing or "none"); return
    if a.only:
        models = [m for m in models if m["label"] in a.only]

    pc = PromptConfFactory(os.path.join(a.llmmap, "confs/prompt_configurations"))
    queries = json.load(open(os.path.join(a.llmmap, "confs/queries/default.json")))
    train = sample_confs(pc, a.n_train, TRAIN, a.seed)
    test = sample_confs(pc, a.n_test, TEST, a.seed + 1)
    os.makedirs(a.out, exist_ok=True)

    for m in models:
        if m["tag"] in missing:
            print("skip (not pulled):", m["tag"]); continue
        path = os.path.join(a.out, m["label"] + a.think + ".jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path):
                e = json.loads(line); done.add((e["dataset"], e["conf_id"]))
        llm = LLM_ollama(m["label"], m["tag"], reasoning=m.get("reasoning", False), think=a.think)
        jobs = [(TRAIN, i, c) for i, c in enumerate(train)] + [(TEST, i, c) for i, c in enumerate(test)]
        jobs = [j for j in jobs if (j[0], j[1]) not in done]

        def run(job):
            pool, i, conf = job
            traces = []
            for q in queries:
                prompt, hp = conf(q, llm)
                traces.append((q, llm.generate(prompt, dict(hp))[0]))
            return {"dataset": pool, "conf_id": i, "llm": m["label"], "traces": traces,
                    "prompt_conf": conf.to_dict()}

        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
            for entry in tqdm.tqdm(ex.map(run, jobs), total=len(jobs), desc=m["label"]):
                f.write(json.dumps(entry) + "\n"); f.flush()

if __name__ == "__main__":
    main()

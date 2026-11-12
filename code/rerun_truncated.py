"""
Rerun the unparsed (-2) runs of a v3 fingerprint with a larger token budget.

Why: thinking models (qwen3:4b, deepseek-r1:8b) often spent the whole num_predict=4096 budget
thinking and returned an empty answer, which was recorded as unparsed (-2). This reruns ONLY
those (index, run) slots with num_predict=8192 (num_ctx=10240); everything else (prompt,
Ollama default sampling, parser) is identical to generate_predictions.py.

The original v3 file is never modified. Outputs:
  fingerprints_v3_rerun/<col>.rerun.jsonl   one line per rerun slot:
      {"index", "run", "choice", "done_reason", "eval_count", "tail"}   (resumable; -1 slots retried)
  fingerprints_v3_rerun/<col>.csv.jsonl     merged copy of the v3 jsonl (same format), with every
                                            rerun -2 slot replaced by its new choice
  fingerprints_v3_rerun/<col>.csv           merged CSV, same columns as generate_predictions.py

Note: the jsonl does not record WHY a run was unparsed, so all -2 slots are rerun (for qwen3:4b
1976/1978 and deepseek-r1:8b 1152/1199 were empty, i.e. budget exhaustion).

Usage: python rerun_truncated.py --model qwen3:4b [--dry-run]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
import generate_predictions as gp

NUM_PREDICT = 8192
NUM_CTX = 10240
TIMEOUT = 3600
SRC_DIR = "fingerprints_v3"
OUT_DIR = "fingerprints_v3_rerun"


def query(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}}
    r = gp._session().post(f"{config.OLLAMA_HOST}/api/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def one(model, idx, run, prompt):
    for attempt in range(config.MAX_RETRIES):
        try:
            j = query(model, prompt)
        except Exception as e:
            print(f"[{model}] {idx}/{run} attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        text = j.get("response", "")
        return {"index": idx, "run": run, "choice": gp.parse_choice(text),
                "done_reason": j.get("done_reason"), "eval_count": j.get("eval_count"),
                "tail": text[-300:]}
    return {"index": idx, "run": run, "choice": gp.FAILED, "done_reason": None,
            "eval_count": None, "tail": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    col = config.col_name(a.model)
    src = f"{SRC_DIR}/{col}.csv.jsonl"
    os.makedirs(OUT_DIR, exist_ok=True)
    rerun_path = f"{OUT_DIR}/{col}.rerun.jsonl"

    df = pd.read_csv(config.DATASET_PATH)
    last = {}
    for line in open(src):
        r = json.loads(line)
        last[r["index"]] = r["choices"]
    complete = {i: c for i, c in last.items() if len(c) == a.n_runs and gp.FAILED not in c}
    assert len(complete) == len(df), f"{src}: only {len(complete)}/{len(df)} rows complete; finish v3 first"
    slots = [(i, k) for i, c in complete.items() for k, v in enumerate(c) if v == gp.UNPARSED]

    done = {}
    if os.path.exists(rerun_path):
        for line in open(rerun_path):
            r = json.loads(line)
            if r["choice"] != gp.FAILED:
                done[(r["index"], r["run"])] = r
    todo = [s for s in slots if s not in done]
    print(f"[{a.model}] unparsed slots={len(slots)} already rerun={len(done)} todo={len(todo)} "
          f"num_predict={NUM_PREDICT} num_ctx={NUM_CTX}", flush=True)
    if a.dry_run:
        return

    rows = df.set_index("index")
    par = int(os.environ.get("OLLAMA_NUM_PARALLEL", "10"))
    t0, n = time.time(), 0
    with open(rerun_path, "a") as log, ThreadPoolExecutor(max_workers=par) as ex:
        futs = []
        for idx, run in todo:
            row = rows.loc[idx]
            prompt = gp.build_prompt(row["ctx"], [row[f"end{i}"] for i in range(4)])
            futs.append(ex.submit(one, a.model, idx, run, prompt))
        for fu in as_completed(futs):
            r = fu.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            n += 1
            if r["choice"] != gp.FAILED:
                done[(r["index"], r["run"])] = r
            if n % 100 == 0 or n == len(todo):
                print(f"[{a.model}] rerun {len(done)}/{len(slots)} | {n / max(time.time() - t0, 1e-9) * 3600:.0f} runs/h",
                      flush=True)

    # merge
    merged = {i: list(c) for i, c in complete.items()}
    for (i, k), r in done.items():
        merged[i][k] = r["choice"]
    with open(f"{OUT_DIR}/{col}.csv.jsonl", "w") as f:
        for i in df["index"]:
            f.write(json.dumps({"index": int(i), "choices": merged[int(i)]}) + "\n")

    out = df[["index", "ctx", "label", "end0", "end1", "end2", "end3"]].copy()
    run_cols = [f"{col}_run{k}" for k in range(a.n_runs)]
    out = pd.concat([out, pd.DataFrame([merged[int(i)] for i in out["index"]], columns=run_cols,
                                       index=out.index).astype("Int64")], axis=1)
    out[f"{col}_mode"] = pd.array([gp.mode_of(merged[int(i)]) for i in out["index"]], dtype="Int64")
    out[f"{col}_mode_match"] = out[f"{col}_mode"] == out["label"]
    out.to_csv(f"{OUT_DIR}/{col}.csv", index=False)

    vals = [c for v in merged.values() for c in v]
    rr = list(done.values())
    still_len = sum(1 for r in rr if r["done_reason"] == "length")
    print(f"[{a.model}] RERUN FINAL slots={len(slots)} rerun_ok={len(done)} "
          f"now_parsed={sum(1 for r in rr if r['choice'] >= 0)} still_unparsed={sum(1 for r in rr if r['choice'] == gp.UNPARSED)} "
          f"hit_8192_limit={still_len} | merged unparsed={vals.count(gp.UNPARSED)}/{len(vals)} "
          f"mode accuracy: {out[f'{col}_mode_match'].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()

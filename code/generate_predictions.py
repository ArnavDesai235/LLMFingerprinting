"""
Stage 1: Generation (resumable, no sharding).

--mode think   : normal reasoning; answer read from `response` (thinking goes to separate field)
--mode nothink : raw prompt with empty <think></think> prefilled -> model answers directly (DeepSeek-R1 only)

Sampling uses each model's Ollama defaults (no temperature/seed override).

Choice codes:  0-3 = answer,  -2 = model output had no unambiguous answer,  -1 = request failed (infra)

Usage:
    python generate_predictions.py --model deepseek-r1:7b --out fingerprints/deepseek-r1_7b.csv --mode think
    python generate_predictions.py --model deepseek-r1:7b --out test_out/x.csv --mode nothink --num-rows 5
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config

FAILED = -1
UNPARSED = -2

# DeepSeek-R1 chat format with reasoning skipped (BOS added by tokenizer)
NOTHINK_TEMPLATE = "<｜User｜>{prompt}<｜Assistant｜><think>\n\n</think>\n\n"

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def build_prompt(ctx: str, ends: list[str]) -> str:
    options = "\n".join(f"{i}: {e}" for i, e in enumerate(ends))
    return (
        f"Context: {ctx}\n\n"
        f"Which ending best continues this? Reply with ONLY the number.\n"
        f"{options}\n\nAnswer:"
    )


def parse_choice(text: str) -> int:
    # drop closed or unclosed think blocks
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    text = re.sub(r"\[THINK\].*?(\[/THINK\]|$)", "", text, flags=re.S).strip()
    m = re.findall(r"[Aa]nswer\W*([0-3])\b", text)
    if m:
        return int(m[-1])
    m = re.fullmatch(r"\W*([0-3])\b.*", text, flags=re.S)       # leads with the number
    if m:
        return int(m.group(1))
    found = set(re.findall(r"\b([0-3])\b", text))
    return int(found.pop()) if len(found) == 1 else UNPARSED      # ambiguous -> unparsed


def query_ollama(model: str, prompt: str, mode: str) -> str:
    if mode == "nothink":
        payload = {
            "model": model,
            "prompt": NOTHINK_TEMPLATE.format(prompt=prompt),
            "raw": True,
            "stream": False,
            "options": {"num_predict": 16},
        }
        timeout = 60
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 4096, "num_ctx": 8192},
        }
        timeout = 900   # 4096 tokens with 10 concurrent streams can exceed 180s
    resp = _session().post(f"{config.OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


def query_once_with_retries(model: str, prompt: str, mode: str) -> int:
    for attempt in range(config.MAX_RETRIES):
        try:
            text = query_ollama(model, prompt, mode)
        except Exception as e:
            print(f"[{model}] attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        choice = parse_choice(text)
        if choice == UNPARSED:
            print(f"[{model}] unparsed: {text[-200:]!r}", file=sys.stderr)
        return choice
    return FAILED


def mode_of(choices: list[int]) -> int:
    """Most common valid answer; ties -> earliest run. Failure codes only if every run failed."""
    valid = [c for c in choices if c >= 0]
    return Counter(valid or choices).most_common(1)[0][0]


def run_model(model: str, df: pd.DataFrame, out_path: str, n_runs: int, mode: str) -> pd.DataFrame:
    col = config.col_name(model)
    log_path = out_path + ".jsonl"
    parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", "10"))

    # resume: rows with any infra failure are redone; later lines override earlier ones
    done = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                r = json.loads(line)
                if len(r["choices"]) == n_runs and FAILED not in r["choices"]:
                    done[r["index"]] = r["choices"]
                else:
                    done.pop(r["index"], None)
        print(f"[{model}] resuming from {log_path}: {len(done)} rows complete")
    else:
        print(f"[{model}] no existing output found, starting fresh")

    todo = df[~df["index"].isin(done)]
    pending = {}
    t0 = time.time()
    n_new = 0

    def task(idx, prompt, run):
        return idx, run, query_once_with_retries(model, prompt, mode)

    # one pool sized to Ollama's slots: the GPU never idles waiting on a row's slowest run,
    # and no request waits in Ollama's queue while its timeout runs
    with open(log_path, "a") as log, ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = []
        for _, row in todo.iterrows():
            idx = int(row["index"])
            prompt = build_prompt(row["ctx"], [row[f"end{i}"] for i in range(4)])
            pending[idx] = [None] * n_runs
            futs += [ex.submit(task, idx, prompt, r) for r in range(n_runs)]

        for fu in as_completed(futs):
            idx, run, choice = fu.result()
            pending[idx][run] = choice
            if None in pending[idx]:
                continue
            choices = pending.pop(idx)
            log.write(json.dumps({"index": idx, "choices": choices}) + "\n")
            log.flush()
            n_new += 1
            if FAILED not in choices:
                done[idx] = choices
            if n_new % 100 == 0 or not pending:
                rate = n_new / max(time.time() - t0, 1e-9) * 3600
                print(f"[{model}|{mode}] {len(done)}/{len(df)} rows | {rate:.0f} rows/h", flush=True)

    out = df[["index", "ctx", "label", "end0", "end1", "end2", "end3"]].copy()
    run_cols = [f"{col}_run{i}" for i in range(n_runs)]
    runs = pd.DataFrame([done.get(int(i), [pd.NA] * n_runs) for i in out["index"]],
                        columns=run_cols, index=out.index).astype("Int64")
    out = pd.concat([out, runs], axis=1)
    out[f"{col}_mode"] = pd.array(
        [mode_of(done[int(i)]) if int(i) in done else pd.NA for i in out["index"]], dtype="Int64")
    out[f"{col}_mode_match"] = out[f"{col}_mode"] == out["label"]
    out.to_csv(out_path, index=False)

    vals = [c for v in done.values() for c in v]
    missing = len(df) - len(done)
    print(f"[{model}|{mode}] FINAL rows={len(done)}/{len(df)} missing={missing} "
          f"unparsed={vals.count(UNPARSED)}/{len(vals)} "
          f"mode accuracy: {out[f'{col}_mode_match'].mean():.4f}")
    if missing:
        print(f"[{model}] {missing} rows had request failures; rerun the same command to retry them",
              file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--num-rows", type=int, default=None)
    ap.add_argument("--mode", choices=["think", "nothink"], default="think")
    args = ap.parse_args()

    df = pd.read_csv(config.DATASET_PATH)
    assert not df["index"].duplicated().any(), "duplicate index values in dataset"
    n = args.num_rows if args.num_rows is not None else config.NUM_ROWS
    if n is not None:
        df = df.head(n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"dataset: {len(df)} rows, {args.n_runs} runs/row, mode={args.mode}")
    run_model(args.model, df, args.out, args.n_runs, args.mode)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

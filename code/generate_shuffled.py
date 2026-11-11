"""
Shuffled-options generation (resumable). Same prompt, parsing and sampling as generate_predictions.py
(imported from it), but every run shows the four endings in a different order.

Order: perm = permutation of [0,1,2,3] drawn from numpy default_rng([SEED, index, run]), so it depends only
on (question, run): every model sees exactly the same 10 orders for a question. Displayed slot s shows
original ending perm[s].

Output: <out>.jsonl, one line per finished row:
  {"index": i, "choices": [10 x original-ending id 0-3, or -2 unparsed / -1 failed],
   "slots":   [10 x displayed slot the model answered, same codes],
   "perms":   [10 x [4 ints]]}
"choices" is in the same coordinates as the unshuffled runs (compare with label/end0-3 directly);
"slots" is the raw position, for position-bias analysis.

Usage:
    python generate_shuffled.py --model qwen2.5:7b --out fingerprints_shuffle/qwen2.5_7b.csv
    python generate_shuffled.py --model qwen2.5:7b --out test_out/x.csv --num-rows 3 --n-runs 2
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config
from generate_predictions import FAILED, UNPARSED, build_prompt, query_once_with_retries

SEED = 20261111


def perm_for(idx: int, run: int) -> list[int]:
    return [int(x) for x in np.random.default_rng([SEED, idx, run]).permutation(4)]


def run_model(model, df, out_path, n_runs, mode):
    log_path = out_path + ".jsonl"
    parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", "10"))

    done = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                r = json.loads(line)
                if len(r["choices"]) == n_runs and FAILED not in r["choices"]:
                    done[r["index"]] = r
                else:
                    done.pop(r["index"], None)
        print(f"[{model}] resuming from {log_path}: {len(done)} rows complete")
    else:
        print(f"[{model}] no existing output found, starting fresh")

    todo = df[~df["index"].isin(done)]
    pending, t0, n_new = {}, time.time(), 0

    def task(idx, run, prompt):
        return idx, run, query_once_with_retries(model, prompt, mode)

    with open(log_path, "a") as log, ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = []
        for _, row in todo.iterrows():
            idx = int(row["index"])
            ends = [row[f"end{i}"] for i in range(4)]
            perms = [perm_for(idx, r) for r in range(n_runs)]
            pending[idx] = {"perms": perms, "slots": [None] * n_runs}
            futs += [ex.submit(task, idx, r, build_prompt(row["ctx"], [ends[p] for p in perms[r]]))
                     for r in range(n_runs)]

        for fu in as_completed(futs):
            idx, run, slot = fu.result()
            pending[idx]["slots"][run] = slot
            if None in pending[idx]["slots"]:
                continue
            p = pending.pop(idx)
            choices = [p["perms"][r][s] if s >= 0 else s for r, s in enumerate(p["slots"])]
            rec = {"index": idx, "choices": choices, "slots": p["slots"], "perms": p["perms"]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            n_new += 1
            if FAILED not in choices:
                done[idx] = rec
            if n_new % 50 == 0 or not pending:
                rate = n_new / max(time.time() - t0, 1e-9) * 3600
                print(f"[{model}|shuffle] {len(done)}/{len(df)} rows | {rate:.0f} rows/h", flush=True)

    lab = dict(zip(df["index"], df["label"]))
    vals = [c for r in done.values() for c in r["choices"]]
    modes = [max(set(v := [c for c in r["choices"] if c >= 0] or r["choices"]), key=v.count) for r in done.values()]
    acc = np.mean([m == lab[i] for m, i in zip(modes, done)]) if done else float("nan")
    missing = len(df) - len(done)
    print(f"[{model}|shuffle] FINAL rows={len(done)}/{len(df)} missing={missing} "
          f"unparsed={vals.count(UNPARSED)}/{len(vals)} mode accuracy: {acc:.4f}")


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
    if args.num_rows is not None:
        df = df.head(args.num_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"dataset: {config.DATASET_PATH} {len(df)} rows, {args.n_runs} runs/row, shuffled options, seed {SEED}")
    run_model(args.model, df, args.out, args.n_runs, args.mode)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

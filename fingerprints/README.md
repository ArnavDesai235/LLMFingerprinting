# Fingerprints

There is one file per model, `<model with ':' replaced by '_'>.csv.jsonl`, with one JSON line per question. Each model also has a `.params.txt` file holding the Ollama sampling parameters and Ollama version used.

| Directory | Questions | Order | Status (2026-11-11) |
|---|---|---|---|
| `v3_clean3k/` | `data/hellaswag_val_clean_3k.csv` | fixed | 28/30 models complete. `qwen3_4b` has 2,795/3,000 rows; `qwen3:1.7b` has not started |
| `holdout500/` | `data/hellaswag_val_clean_holdout500.csv` | fixed | 30/30 complete |
| `shuffle500/` | same 500 held-out questions | new order every run | 22/30 complete. `deepseek-r1_8b` and `phi3_3.8b` are partial; the rest are still running |

## Format

```json
{"index": 8156, "choices": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}
```

`choices` holds the 10 independent runs. Codes: `0-3` = chosen ending, `-2` = no unambiguous answer in the output, `-1` = request failed.

The shuffled runs carry extra fields:

```json
{"index": 2083, "choices": [3, 2, ...], "slots": [0, 3, ...], "perms": [[3, 0, 1, 2], ...]}
```

- `choices` is already mapped back to the original ending ids, so it can be compared with `label` and with the unshuffled runs directly.
- `slots` is the displayed position the model chose.
- `perms[r][s]` is the original ending shown in slot `s` on run `r`. It is seeded by `(20261111, index, run)`, so every model sees the same 10 orders.

## Notes

- `v3_clean3k/phi3_3.8b.csv.jsonl` has 3,013 lines for 3,000 questions, because a few rows were re-written on resume. The loaders key rows by `index` (the last line wins). phi3:3.8b also logged 351 failed queries (timeouts, then retried), because on this data it tends to ramble.
- Analysis protocol everywhere: runs 0-4 build the reference fingerprint and runs 5-9 are used as probes.

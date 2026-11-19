# Fingerprints

There is one file per model, `<model with ':' replaced by '_'>.csv.jsonl`, with one JSON line per question. Each model also has a `.params.txt` file holding the Ollama sampling parameters and Ollama version used.

| Directory | Questions | Order | Status (2026-11-19) |
|---|---|---|---|
| `v3_clean3k/` | `data/hellaswag_val_clean_3k.csv` | fixed | 30/30 complete (first run, 4096-token budget) |
| `v3_clean3k_rerun8192/` | same | fixed | deepseek-r1_8b, qwen3_4b, qwen3_1.7b with every unparsed run redone at num_predict 8192 (merged files). These replace the first-run files in the main analyses |
| `variants3k/` | same | fixed | 14/14 complete: 10 abliterated models and 4 reasoning fine-tunes (see below) |
| `holdout500/` | `data/hellaswag_val_clean_holdout500.csv` | fixed | 30/30 complete |
| `shuffle500/` | same 500 held-out questions | new order every run | 30/30 complete |

## Variants (`variants3k/`)

File names keep the Ollama name (`/` -> `__`, `:` -> `_`). `analysis/main/labels.py` maps each one to a library name, its parent and its kind.

| File stem | Parent (base model) | Kind | Note |
|---|---|---|---|
| huihui_ai__falcon3-abliterated_10b / _7b / _3b | falcon3 10b / 7b / 3b | abliterated | |
| huihui_ai__deepseek-r1-abliterated_14b-qwen-distill / _7b-qwen-distill | deepseek-r1 14b / 7b | abliterated | |
| huihui_ai__deepseek-r1-abliterated_8b-llama-distill | none in the library | abliterated | Llama-3.1-8B distill; Ollama's deepseek-r1:8b is the Qwen3-8B distill (R1-0528) |
| huihui_ai__llama3.2-abliterate_3b | llama3.2 3b | abliterated | |
| mannix__llama3.1-8b-abliterated | llama3.1 8b | abliterated | Q4_0 (parent Q4_K_M) |
| krith__mistral-nemo-instruct-2407-abliterated_IQ4_XS | mistral-nemo 12b | abliterated | IQ4_XS (parent Q4_0) |
| richardyoung__mistral-7b-instruct-v0.3-abliterated_Q4_K_M | mistral 7b | abliterated | |
| openthinker_7b, smallthinker_3b | qwen2.5 7b, 3b | reasoning SFT | smallthinker run with FP_BOXED=1 (\boxed answers), Q8_0 |
| phi4-reasoning_14b, phi4-mini-reasoning_3.8b | phi4 14b, phi4-mini 3.8b | reasoning SFT | |

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

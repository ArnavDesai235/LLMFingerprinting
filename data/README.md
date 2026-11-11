# Data

Clean samples of the HellaSwag **validation** split ([Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)).

| File | Rows | Contents |
|---|---|---|
| `hellaswag_val_clean_3k.csv` | 3,000 | Main set. Stratified sample, seed 0: 2,031 WikiHow and 969 ActivityNet (ActivityNet rows listed first) |
| `hellaswag_val_clean_holdout500.csv` | 500 | Held-out set: 500 validation rows not in the 3k, with the same stratification (338 WikiHow and 162 ActivityNet, seed 1). Made by `code/make_holdout.py` |

Columns: `index` (row id, used as the key in all fingerprint files), `ctx`, `label` (0-3), `end0`..`end3`, `orig_ind`, `source_id`, `split_type`, `stripped_endings_intact`.

The full clean validation set (`hellaswag_val_clean_reordered.csv`, 10,042 rows, 8.5 MB) is not in the repo. It is on the cluster under `~/APfinalwork/data_clean/`.

## Why "clean"?

The earlier file `reorderedhellaswag.csv` was corrupted. In every WikiHow row (68% of the data), the endings were shifted between columns, so `label` pointed at the wrong text. ActivityNet rows were intact. All v2 results (see `legacy/`) and the CSE523 report were computed on that file. Everything in this repo outside `legacy/` uses the clean files.

# Legacy: v2 fingerprints (corrupted dataset). Do not use

The CSVs in `fingerprints_v2_corrupted/` were produced from `reorderedhellaswag.csv`. All of its WikiHow rows (68%) have endings shifted between columns, so the label points at the wrong text. Accuracy and anything that depends on which answer is wrong are invalid. Behavioural similarity between models (answer agreement) is still meaningful, but use the clean v3 data in `fingerprints/` for everything.

Format: one row per question, with columns `run0..run9` (choice codes as in `fingerprints/README.md`), the per-row mode, and `mode_match`, over all 10,042 validation rows.

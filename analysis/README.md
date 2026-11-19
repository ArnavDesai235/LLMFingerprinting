# Analysis

All scripts read the fingerprint `.jsonl` files and use the same protocol: runs 0-4 build the reference and runs 5-9 are the probes. Choice codes outside 0-3 are folded into one "unparsed or failed" option.

## `final/`: final runs on the complete data (Nov 19)

```bash
bash analysis/final/run_all.sh
```

This builds three model sets with `final/build_sets.py` (base30 with the 8192-token reruns, base30_raw, and all44 = base30 + 14 variants). It runs every script below on them and writes `results/logs/final_nov19/<set>/` and `results/pairs/final_nov19/<set>/`, then `make_figures.py`. It takes about 6 minutes. `results/logs/final_nov19/README.md` lists the changes and the headline numbers. Model labels (family, lineage, reasoning, variant parent) live in `main/labels.py`.

| Script | Question |
|---|---|
| `main/a19_variants.py <base_dir> <variants_dir> <csv>` | Do fingerprints survive abliteration and reasoning fine-tuning? (variant vs parent similarity, identification, open set, drift) |
| `main/a20_library_with_variants.py` | Identification when the library also holds the variants (twin confusion) |

## `make_figures.py`

```bash
python analysis/make_figures.py
```

This regenerates every figure in `figures/` and every table in `results/tables/` from the current `fingerprints/` (about 15 s). Fig. 4 reads `results/pairs/pairs_lineage_reasoning.csv` (written by `main/a14_lineage_vs_reasoning.py`).

## `main/`: current analyses (clean v3, Nov 9-11)

`load.py` defaults to `fingerprints/v3_clean3k` and `data/hellaswag_val_clean_3k.csv`. Override these with `FPV3=<dir>` and `FPCSV=<csv>`. Run scripts that import `load` from inside `analysis/main/`.

| Script | Question | Output in `results/logs/` |
|---|---|---|
| `../rq1_analysis.py <fp_dir> <csv>` | Identification vs k (all, errors-only, correct-only), MI question selection, open set, size vs family regressions | `rq1_23.txt`, `rq1_26.txt` |
| `a13_distill.py` | Do the DeepSeek-R1 distills share errors with their base model, or with each other? | `a13.txt`, `a13b.txt` |
| `a14_lineage_vs_reasoning.py` | Accuracy-controlled OLS over all pairs: same base lineage vs both reasoning-trained, with model-level permutation p-values | `a14.txt` (+ `pairs/pairs_lineage_reasoning.csv`) |
| `a15_corr_error_id.py` | Identification with correlated-error distances (conditional Hamming, co-error) vs likelihood | `a15.txt` |
| `a16_unseen_questions.py` | Does a fingerprint transfer to questions it has not seen? (text-feature conditional logit) | `a16.txt` |
| `a17_holdout.py <v3_dir> <v3_csv> <ho_dir> <ho_csv>` | Held-out replication: identification, frozen open-set thresholds, stability, family and reasoning effects | `a17_holdout*.txt` |
| `a18_shuffle.py <ho_dir> <shuf_dir> <ho_csv>` | Shuffled options: content vs slot, accuracy, error persistence, identification fixed vs shuffled, family effect | `a18_shuffle.txt` |
| `a18b_family.py` (same args as a18) | Per-family breakdown, fixed vs shuffled | `a18b_family.txt` |

The `a12_*.txt` logs are `early_v3/a12_family_17models.py` re-run on the 23- and 26-model snapshots.

## `early_v3/`: first-pass analyses (12-17 models, Nov 7)

| Script | Topic |
|---|---|
| `a1_profile.py` | Accuracy, unanimity, position bias. Also writes the `A_full.npy`, `y.npy` and `full.csv` caches that later scripts load, so **run it first** |
| `a2_identification.py`, `a4_open_set.py`, `a5_partial_models.py` | Identification vs k, open set, partial models |
| `a3_error_correlation.py`, `a6_claim_checks.py` | Error correlation, where vs which wrong, MI with target held out, position-bias null |
| `a7_report_metrics.py`, `a8_overlap_stats.py`, `a9_v2_hamming_vs_jaccard.py` | Re-check of the CSE523 report metrics (D_H vs Jaccard) and the Oct 8 Mistral overlap note |
| `a10_likelihood_vs_mode.py`, `a11_condhamming_decomposition.py` | Likelihood vs mode fingerprint; D_H = 1 − P(a wrong given b wrong) × P(same wrong given both wrong) |
| `a12_family_17models.py` | Accuracy-controlled family test, per-family breakdown |

## `fingerprint_analysis.py`

The original v2 analysis (profile, identification vs k, similarity). Kept for reference.

#!/usr/bin/env bash
# Final rerun of every analysis on the complete data (2026-11-19).
# Usage (from the repo root): bash analysis/final/run_all.sh [work_dir]
# Writes logs to results/logs/final_nov19/<set>/ and pair tables to results/pairs/final_nov19/<set>/.
#   base30      30 base models, 3k questions, 8192-token reruns merged (main numbers)
#   all44       base30 + 14 abliterated / reasoning-fine-tuned variants
#   base30_raw  30 base models, first-run data (4096 tokens), used for 3k-vs-held-out (a17)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-$(mktemp -d)}
CSV3=$ROOT/data/hellaswag_val_clean_3k.csv
CSVH=$ROOT/data/hellaswag_val_clean_holdout500.csv
LOGS=$ROOT/results/logs/final_nov19; PAIRS=$ROOT/results/pairs/final_nov19
python "$ROOT/analysis/final/build_sets.py" "$WORK"

run_set() {  # $1 = set name
  local S=$1 R=$WORK/run_$1; mkdir -p "$R" "$LOGS/$S" "$PAIRS/$S"
  cp "$ROOT"/analysis/main/{load,labels,a13_distill,a14_lineage_vs_reasoning,a15_corr_error_id,a16_unseen_questions}.py "$R/"
  cp "$ROOT"/analysis/early_v3/{a1_profile,a12_family_17models}.py "$R/"
  export FPV3=$WORK/$S FPCSV=$CSV3
  (cd "$R" && python "$ROOT/analysis/rq1_analysis.py" "$WORK/$S" "$CSV3" > "$LOGS/$S/rq1.txt" && mv pairwise_metrics_v3.csv "$PAIRS/$S/")
  (cd "$R" && python a1_profile.py > "$LOGS/$S/a1_profile.txt" && python a12_family_17models.py > "$LOGS/$S/a12_family.txt" \
     && mv pairs_family_17.csv "$PAIRS/$S/pairs_family.csv")
  (cd "$R" && python a13_distill.py > "$LOGS/$S/a13_distill.txt" && mv pairs_distill.csv "$PAIRS/$S/")
  (cd "$R" && python a14_lineage_vs_reasoning.py > "$LOGS/$S/a14_lineage_vs_reasoning.txt" && mv pairs_lineage_reasoning.csv "$PAIRS/$S/")
  (cd "$R" && python a15_corr_error_id.py > "$LOGS/$S/a15_corr_error_id.txt")
  (cd "$R" && python a16_unseen_questions.py > "$LOGS/$S/a16_unseen_questions.txt")
  echo "done $S"
}
run_set base30 &
( run_set all44 && cd "$WORK/run_all44" && cp "$ROOT/analysis/main/a20_library_with_variants.py" . \
  && FPV3=$WORK/all44 FPCSV=$CSV3 python a20_library_with_variants.py > "$LOGS/all44/a20_library_with_variants.txt" ) &
( mkdir -p "$LOGS/base30_raw" "$LOGS/base30" && cd "$ROOT/analysis/main" \
  && python a17_holdout.py "$WORK/base30_raw" "$CSV3" "$ROOT/fingerprints/holdout500" "$CSVH" > "$LOGS/base30_raw/a17_holdout.txt" \
  && python a18_shuffle.py "$ROOT/fingerprints/holdout500" "$ROOT/fingerprints/shuffle500" "$CSVH" > "$LOGS/base30/a18_shuffle.txt" \
  && python a18b_family.py "$ROOT/fingerprints/holdout500" "$ROOT/fingerprints/shuffle500" "$CSVH" > "$LOGS/base30/a18b_family.txt" \
  && python a19_variants.py "$WORK/base30" "$ROOT/fingerprints/variants3k" "$CSV3" "$PAIRS/a19" > "$LOGS/base30/a19_variants.txt" \
  && echo "done a17-a19" ) &
wait
cp "$PAIRS/base30/pairs_lineage_reasoning.csv" "$ROOT/results/pairs/pairs_lineage_reasoning.csv"   # read by make_figures (fig 4)
python "$ROOT/analysis/make_figures.py" > "$LOGS/make_figures.txt"
echo "all done: $LOGS"

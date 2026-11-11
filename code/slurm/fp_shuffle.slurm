#!/bin/bash
#SBATCH --job-name=fp_shuf
#SBATCH --partition=defq
#SBATCH --chdir=/home/armdesai/APfinalwork
#SBATCH --output=logs/fpshuf_%x_%j.out
#SBATCH --error=logs/fpshuf_%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=h100
# usage: sbatch -J shuf_<name> fp_shuffle.slurm <model> [model ...]
# Shuffled-options runs (generate_shuffled.py): new option order every run, same orders for every model.
# Default dataset = the 500 held-out questions. Resubmitting resumes where it stopped.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ollama
export FP_DATASET=${FP_DATASET:-data_clean/hellaswag_val_clean_holdout500.csv}
echo "dataset=$FP_DATASET"

MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || { echo "usage: sbatch fp_shuffle.slurm <model> [model ...]"; exit 1; }
NAME="${1//:/_}"
OUTDIR=fingerprints_shuffle
LOGDIR="logs/fpshuf_${NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR" "$LOGDIR"

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_NUM_PARALLEL=10
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES host=$OLLAMA_HOST models=${MODELS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# drop the per-token 'print_timing' lines: they made earlier serve logs 2.5-9.6 GB
ollama serve > >(grep --line-buffered -v print_timing > "$LOGDIR/ollama_serve.log") 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "server failed"; cat "$LOGDIR/ollama_serve.log"; exit 1; }
grep "inference compute" "$LOGDIR/ollama_serve.log"

for m in "${MODELS[@]}"; do
  col=$(python3 -c "import config; print(config.col_name('$m'))")
  out="$OUTDIR/${col}.csv"
  echo "[$(date '+%F %T')] ===== $m -> $out ====="
  { ollama show "$m" --parameters; ollama --version; } > "$OUTDIR/${col}.params.txt" 2>&1
  python3 -u generate_shuffled.py --model "$m" --out "$out" --n-runs 10 --mode think \
      > "$LOGDIR/${col}.log" 2>&1
  echo "[$(date '+%F %T')] done $m: $(grep FINAL "$LOGDIR/${col}.log" | tail -1)"
done

echo "[$(date '+%F %T')] shuffle job $NAME finished"

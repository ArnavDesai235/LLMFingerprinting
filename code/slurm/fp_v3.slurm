#!/bin/bash
#SBATCH --job-name=fp_fam
#SBATCH --partition=defq
#SBATCH --chdir=/home/armdesai/APfinalwork
#SBATCH --output=logs/fpv3_%x_%j.out
#SBATCH --error=logs/fpv3_%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=h100
# usage: sbatch -J fp_<family> fp_family.slurm <deepseek|gemma|llama|phi>
#    or: sbatch -J fp_<name> fp_family.slurm <model> [model ...]
# One GPU, no sharding: runs every model in the family on the full dataset, one after another.
# Sampling uses each model's Ollama defaults. Resubmitting resumes where it stopped.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ollama
export FP_DATASET=${FP_DATASET:-data_clean/hellaswag_val_clean_3k.csv}   # clean HellaSwag (v3)
echo "dataset=$FP_DATASET"

FAMILY=${1:?usage: sbatch -J fp_<family> fp_family.slurm <deepseek|gemma|llama|phi>}
case "$FAMILY" in
  deepseek) MODELS=("deepseek-r1:14b" "deepseek-r1:8b" "deepseek-r1:7b" "deepseek-r1:1.5b") ;;
  gemma)    MODELS=("gemma3:27b" "gemma3:12b" "gemma3:4b" "gemma3:1b") ;;
  llama)    MODELS=("llama3.1:8b" "llama3.2:3b" "llama3.2:1b") ;;
  phi)      MODELS=("phi4:14b" "phi3:14b" "phi4-mini:3.8b" "phi3:3.8b") ;;
  *) MODELS=("$@"); FAMILY="${1//:/_}" ;;   # or pass model names directly
esac

OUTDIR=fingerprints_v3
LOGDIR="logs/fpv3_${FAMILY}_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR" "$LOGDIR"

# free port, so several family jobs can share a node
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_NUM_PARALLEL=10
echo "family=$FAMILY node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES host=$OLLAMA_HOST models=${MODELS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

ollama serve > "$LOGDIR/ollama_serve.log" 2>&1 &
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
  python3 -u generate_predictions.py --model "$m" --out "$out" --n-runs 10 --mode think \
      > "$LOGDIR/${col}.log" 2>&1
  echo "[$(date '+%F %T')] done $m: $(grep FINAL "$LOGDIR/${col}.log" | tail -1)"
done

echo "[$(date '+%F %T')] family $FAMILY finished"

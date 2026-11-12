#!/bin/bash
#SBATCH --job-name=v3_rerun
#SBATCH --partition=defq
#SBATCH --chdir=/home/armdesai/APfinalwork
#SBATCH --output=logs/fprerun_%x_%j.out
#SBATCH --error=logs/fprerun_%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=h100
# usage: sbatch -J v3_rerun rerun_truncated.slurm <model> [model ...]
# Reruns the unparsed (-2) runs of fingerprints_v3/<model>.csv.jsonl with num_predict=8192.
# Writes fingerprints_v3_rerun/; never touches fingerprints_v3/. Resubmitting resumes.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ollama
export FP_DATASET=${FP_DATASET:-data_clean/hellaswag_val_clean_3k.csv}
echo "dataset=$FP_DATASET"

MODELS=("$@")
LOGDIR="logs/fprerun_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR" fingerprints_v3_rerun

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_NUM_PARALLEL=10
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES host=$OLLAMA_HOST models=${MODELS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

ollama serve > "$LOGDIR/ollama_serve.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "server failed"; cat "$LOGDIR/ollama_serve.log"; exit 1; }

for m in "${MODELS[@]}"; do
  col=$(python3 -c "import config; print(config.col_name('$m'))")
  echo "[$(date '+%F %T')] ===== rerun $m ====="
  python3 -u rerun_truncated.py --model "$m" > "$LOGDIR/${col}.log" 2>&1
  echo "[$(date '+%F %T')] done $m: $(grep 'RERUN FINAL' "$LOGDIR/${col}.log" | tail -1)"
done
echo "[$(date '+%F %T')] rerun finished"

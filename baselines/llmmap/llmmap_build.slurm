#!/bin/bash
#SBATCH --job-name=llmmap_build
#SBATCH --partition=defq
#SBATCH --output=/home/armdesai/llmmap_repo/baselines/llmmap/logs/%x_%j.out
#SBATCH --error=/home/armdesai/llmmap_repo/baselines/llmmap/logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=h100
# usage: sbatch llmmap_build.slurm [build_dataset.py args]   e.g.  --only qwen2.5_3b --n_train 5 --n_test 2 --out traces_smoke
# Serves Ollama on a free port, then runs build_dataset.py (resumable: resubmit to continue).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ollama
cd /home/armdesai/llmmap_repo/baselines/llmmap; mkdir -p logs
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/ollama_models}
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST="127.0.0.1:$PORT"; export OLLAMA_NUM_PARALLEL=8
echo "node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES host=$OLLAMA_HOST args=$*"
ollama serve > logs/ollama_serve_$SLURM_JOB_ID.log 2>&1 &
trap 'kill $! 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
python3 -u build_dataset.py --check
python3 -u build_dataset.py --llmmap ./LLMmap "$@"
echo "[$(date '+%F %T')] done"

#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=h100
#SBATCH --output=/home/armdesai/llmmap_repo/baselines/llmmap/logs/eval_%j.out
#SBATCH --error=/home/armdesai/llmmap_repo/baselines/llmmap/logs/eval_%j.err
# usage: sbatch llmmap_eval.slurm [evaluate.py args]   (needs HF access once for the e5 embedder)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate llmmap
cd /home/armdesai/llmmap_repo/baselines/llmmap
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
python -c "import torch;print('cuda', torch.cuda.is_available())"
DEV=cpu; python -c "import torch,sys;t=torch.zeros(1).cuda()" 2>/dev/null && DEV=cuda
echo "device=$DEV"
python -u evaluate.py --llmmap ./LLMmap --device $DEV "$@"

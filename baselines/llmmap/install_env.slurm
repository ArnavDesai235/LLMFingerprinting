#!/bin/bash
#SBATCH --job-name=lm_pip
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/armdesai/llmmap_repo/baselines/llmmap/logs/pip_%j.out
# One-time: python env for evaluate.py (conda create -n llmmap python=3.11 first)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate llmmap
pip install torch transformers==4.51.3 scipy numpy requests tqdm anthropic openai pytorch_lightning torchmetrics
python -c "import torch,transformers,scipy;print('OK torch',torch.__version__,'transformers',transformers.__version__)"

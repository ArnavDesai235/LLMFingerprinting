# Generation code

These scripts query models through a local Ollama server, one GPU per SLURM job, with models run one after another. All runs can be resumed: each script skips rows already in the output file.

| File | Purpose |
|---|---|
| `generate_predictions.py` | Main generator. Builds the multiple-choice prompt, queries Ollama with the model's default sampling, parses the letter answer and appends `{"index", "choices"}` lines. `--mode think` (default) or `nothink` (DeepSeek-R1 only, with an empty `<think>` prefilled) |
| `generate_shuffled.py` | Same prompt and parsing, but each run shows the endings in a seeded permutation. Writes `choices`, `slots` and `perms` |
| `make_holdout.py` | Builds the 500-question held-out set from the clean validation pool, excluding the 3k |
| `config.py` | Dataset path (`FP_DATASET`), Ollama host and retries |
| `merge_shards.py` | Helper for older sharded runs |
| `slurm/fp_v3.slurm` | Main job: `sbatch -J <name> fp_v3.slurm <model> [<model> ...]` writes to `fingerprints_v3/`, one GPU, 7-day limit |
| `slurm/fp_holdout.slurm`, `slurm/fp_shuffle.slurm` | The same job for the held-out and shuffled runs |
| `slurm/fp_family.slurm` | Older v2 job script |

The SLURM scripts expect the flat cluster layout: the `.slurm` files, `generate_*.py` and `config.py` all in one directory (`~/APfinalwork`, set by `#SBATCH --chdir`), with the data under `data_clean/`. Copy them there, or edit `--chdir`, before submitting. Each job starts its own `ollama serve` on a free port, saves `ollama show --parameters` to `<model>.params.txt`, and runs each model with `--n-runs 10`.

Example (cluster):

```bash
cd ~/APfinalwork
sbatch -J v3_gemma fp_v3.slurm gemma3:27b gemma3:12b
FP_DATASET=data_clean/hellaswag_val_clean_reordered.csv sbatch -J full_q fp_v3.slurm qwen2.5:7b   # extend to the full set (resumes)
```

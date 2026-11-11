"""
Central config for the fingerprinting pipeline.
Any Ollama model name can be passed to submit_single_model_sized.slurm.
"""
import os


def col_name(model: str) -> str:
    """'llama3.2:latest' -> 'llama3.2_latest' (safe for CSV columns/filenames)."""
    return model.replace(":", "_")


DATASET_PATH = os.environ.get("FP_DATASET", "reorderedhellaswag.csv")   # index,ctx,label,end0,end1,end2,end3
NUM_ROWS = None                             # None = full dataset (override with --num-rows)

# per-job port set by the slurm script (OLLAMA_HOST=127.0.0.1:<port>)
_h = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_HOST = _h if _h.startswith("http") else f"http://{_h}"

MAX_RETRIES = 3

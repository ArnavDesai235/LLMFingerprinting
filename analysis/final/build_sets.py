"""Build the model sets for the final runs (symlinks, nothing copied).

Usage: python analysis/final/build_sets.py <work_dir>
  <work_dir>/base30      30 base models, 3k questions; the 8192-token reruns (fingerprints/v3_clean3k_rerun8192)
                         replace deepseek-r1_8b, qwen3_4b, qwen3_1.7b. Main set.
  <work_dir>/base30_raw  30 base models as first run (4096-token budget for every model), same protocol as the
                         held-out and shuffled runs. Used by a17 (3k vs held-out).
  <work_dir>/all44       base30 + the 14 variants (fingerprints/variants3k), renamed via analysis/main/labels.py.
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'main'))
from labels import VARIANTS

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FP = os.path.join(ROOT, 'fingerprints')
out = os.path.abspath(sys.argv[1])

def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(src, dst)

base = {os.path.basename(f): f for f in glob.glob(f'{FP}/v3_clean3k/*.csv.jsonl')}
assert len(base) == 30, len(base)
for b, f in base.items(): link(f, f'{out}/base30_raw/{b}')
rerun = {os.path.basename(f): f for f in glob.glob(f'{FP}/v3_clean3k_rerun8192/*.csv.jsonl')}
for b, f in {**base, **rerun}.items():
    link(f, f'{out}/base30/{b}'); link(f, f'{out}/all44/{b}')
for stem, (name, parent, kind) in VARIANTS.items():
    link(f'{FP}/variants3k/{stem}.csv.jsonl', f'{out}/all44/{name}.csv.jsonl')
for s in ['base30', 'base30_raw', 'all44']:
    print(s, len(glob.glob(f'{out}/{s}/*.csv.jsonl')), 'models')
print('rerun-merged in base30/all44:', sorted(rerun))

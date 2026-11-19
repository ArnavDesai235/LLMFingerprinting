import sys
import json,glob,os,numpy as np,pandas as pd
import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
D=os.environ.get('FPV3',os.path.join(ROOT,'fingerprints','v3_clean3k'))   # fingerprint .jsonl directory
import sys
CSV=os.environ.get('FPCSV',os.path.join(ROOT,'data','hellaswag_val_clean_3k.csv'))
ds=pd.read_csv(CSV)
def load():
    M={}
    for f in sorted(glob.glob(f'{D}/*.jsonl')):
        m=os.path.basename(f).replace('.csv.jsonl','')
        rows=[json.loads(l) for l in open(f)]
        M[m]={r['index']:r['choices'] for r in rows}
    return M
SIZE={'deepseek-r1_8b':8,'falcon3_10b':10,'falcon3_1b':1,'falcon3_3b':3,'falcon3_7b':7,'gemma3_12b':12,'gemma3_27b':27,'gemma3_4b':4,'mistral-small_22b':22,'mistral-nemo_12b':12,'mistral_7b':7,'gemma3_1b':1,'phi3_14b':14,'phi3_3.8b':3.8,'phi4-mini_3.8b':3.8,'phi4_14b':14,'qwen2.5_1.5b':1.5,'qwen2.5_14b':14,'qwen2.5_3b':3,'qwen2.5_7b':7}
import re
# Nov 19 fix: the old rule (strip digits) split llama3.1 ('llama1') from llama3.2 ('llama2'); now same as main/load.py
def fam(m): return re.match(r'[a-z]+', m).group(0)
def arr(M,models,idx):
    # codes: 0-3 answers, else -> 4 (unparsed/failed)
    A=np.array([[M[m][i] for i in idx] for m in models])
    return np.where((A>=0)&(A<4),A,4)

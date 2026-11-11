"""Does the report's Hamming >> Jaccard gap come from the corrupted labels? Same protocol on v2 (corrupted) repo CSVs."""
import sys,glob,os;sys.path.insert(0,'.')
from load import *
V2='/home/claude/llmfingerprinting'; fs=sorted(glob.glob(f'{V2}/*.csv')); ms=[os.path.basename(f)[:-4] for f in fs]
dfs=[pd.read_csv(f).sort_values('index').reset_index(drop=True) for f in fs]
y=dfs[0].label.to_numpy(); ix=dfs[0]['index'].to_numpy()
A=np.stack([d[[f'{m}_run{r}' for r in range(10)]].to_numpy() for d,m in zip(dfs,ms)]); A=np.where((A>=0)&(A<4),A,4).astype(int)
nm,n,_=A.shape; fams=np.array([fam(m) for m in ms]); rng=np.random.default_rng(9)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
F=cnt(A[:,:,:5])[...,:4].argmax(-1); T=[(b,A[b,:,r]) for b in range(nm) for r in range(5,10)]; truth=np.array([b for b,_ in T])
def D(rows):
    H=np.zeros((len(T),nm));J=np.zeros_like(H); Fe=F[:,rows]!=y[rows]
    for t,(b,R) in enumerate(T):
        Rr=R[rows];Eb=(Rr!=y[rows])&(Rr<4)
        H[t]=[(F[a,rows][Eb]!=Rr[Eb]).mean() for a in range(nm)]
        J[t]=[1-(Fe[a]&Eb).sum()/max((Fe[a]|Eb).sum(),1) for a in range(nm)]
    return H,J
def top1(Dm): return (Dm.argmin(1)==truth).mean()
src=ds.set_index('index').source_id.str.split('~').str[0]
pos={v:i for i,v in enumerate(ix)}
wk=np.array([pos[i] for i in src.index[src=='wikihow']]); an=np.array([pos[i] for i in src.index[src=='activitynet']])
print(f'v2 corrupted, {nm} models, {len(T)} test runs')
for name,pool in [('all 10042 rows',np.arange(n)),('WikiHow (corrupted labels)',wk),('ActivityNet (intact)',an)]:
    for m in [100,500]:
        h=[];j=[]
        for _ in range(10):
            H,J=D(rng.choice(pool,m,replace=False)); h.append(top1(H)); j.append(top1(J))
        print(f'  {name:28s} rows {m:4d}  D_H {np.mean(h):.3f}  D_J {np.mean(j):.3f}')
    H,J=D(pool); print(f'  {name:28s} all {len(pool)} rows  D_H {top1(H):.3f}  D_J {top1(J):.3f}')
# how shared are error sets under corruption?
mo=cnt(A)[...,:4].argmax(-1); W=mo!=y
import itertools
for name,pool in [('WikiHow',wk),('ActivityNet',an)]:
    js=[(W[i,pool]&W[j,pool]).sum()/(W[i,pool]|W[j,pool]).sum() for i,j in itertools.combinations(range(nm),2)]
    print(f'  mean pairwise error-set Jaccard similarity on {name}: {np.mean(js):.3f}; mean error rate {W[:,pool].mean():.3f}')

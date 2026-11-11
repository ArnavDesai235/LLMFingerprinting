import sys;sys.path.insert(0,'.')
from load import *
M=load(); ms=list(M); common=sorted(set.intersection(*[set(M[m]) for m in ms])); print(len(ms),'models, common rows',len(common))
ds2=ds.set_index('index'); y=ds2.loc[common,'label'].to_numpy()
A=arr(M,ms,common); nm,n,_=A.shape; rng=np.random.default_rng(3)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
L=np.log((cnt(A[:,:,:5])+0.5)/(cnt(A[:,:,:5])+0.5).sum(-1,keepdims=True))
mo=cnt(A)[...,:4].argmax(-1)
for i,m in enumerate(ms): 
    if len(M[m])<3000: print(f'  {m:18s} mode_acc {(mo[i]==y).mean():.3f} unparsed {(A[i]==4).mean():.3f} unanimous {(cnt(A[i]).max(1)==10).mean():.2f}')
for k in [5,10,20,50]:
    C=np.zeros((nm,nm),int)
    for _ in range(300):
        for i in range(nm):
            rows=rng.choice(n,k,replace=False); pr=A[i,rows,rng.integers(5,10,k)]
            C[i,L[:,rows,pr].sum(1).argmax()]+=1
    print(k,'exact',(np.trace(C)/C.sum()).round(3))
    if k==10:
        for i,m in enumerate(ms):
            r=C[i].copy();a=r[i]/r.sum();r[i]=-1;j=r.argmax();print(f'    {m:18s} {a:.2f} -> {ms[j]} {C[i,j]/C[i].sum():.2f}')

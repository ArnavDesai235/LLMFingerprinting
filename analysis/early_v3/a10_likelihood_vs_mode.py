"""Likelihood fingerprint vs mode fingerprint on identical probes."""
import sys;sys.path.insert(0,'.')
from load import *
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
def run(A,y,names,ks,trials=400,seed=11,label=''):
    nm,n,_=A.shape; rng=np.random.default_rng(seed)
    C=cnt(A[:,:,:5]); L=np.log((C+0.5)/(C+0.5).sum(-1,keepdims=True)); F=C[...,:4].argmax(-1)
    print(f'== {label}: {nm} models, chance {1/nm:.3f} ==')
    print('   k | likelihood | mode: Hamming all k | mode: cond. Hamming (wrong probes) | ties broken randomly')
    per={}
    for k in ks:
        okL=okH=okC=tot=0; pm=np.zeros((nm,2))
        for _ in range(trials):
            for i in range(nm):
                rows=rng.choice(n,k,replace=False); pr=A[i,rows,rng.integers(5,10,k)]
                sL=L[:,rows,pr].sum(1)
                sH=-(F[:,rows]!=pr).sum(1)+rng.random(nm)*1e-3
                w=(pr!=y[rows])
                sC=-((F[:,rows]!=pr)&w).sum(1)/max(w.sum(),1)+rng.random(nm)*1e-3
                a=sL.argmax()==i; b=sH.argmax()==i
                okL+=a; okH+=b; okC+=sC.argmax()==i; tot+=1; pm[i]+=[a,b]
        print(f'  {k:3d} |   {okL/tot:.3f}    |       {okH/tot:.3f}        |        {okC/tot:.3f}')
        per[k]=pm/trials
    k=10 if 10 in per else ks[0]
    print(f'   per-model at k={k} (likelihood vs mode-Hamming, unanimity):')
    for i,m in enumerate(names):
        print(f'     {m:18s} {per[k][i,0]:.2f} vs {per[k][i,1]:.2f}  gain {per[k][i,0]-per[k][i,1]:+.2f}   unanimous {(C[i].max(-1)==5).mean():.2f}')
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist()
run(A,y,full,[3,5,10,20,50],label='12 complete models, 3000 rows')
M=load(); ms=list(M); common=sorted(set.intersection(*[set(M[m]) for m in ms]))
run(arr(M,ms,common),ds.set_index('index').loc[common,'label'].to_numpy(),ms,[5,10,20],trials=300,label='17 models, 851 ActivityNet rows')

"""Identification accuracy using correlated-error distances vs likelihood, all complete models.
Reference = runs 0-4 (mode for distance methods, counts for likelihood); probe = single answers from runs 5-9.
Methods (nearest neighbour, random tie-break):
  mode_hamming   k random questions; score = # probe answers equal to candidate's mode answer
  cond_hamming   k WRONG probe answers; score = # where candidate's mode is the same wrong ending  (which-wrong, D_H)
  co_error       k WRONG probe answers; score = # where candidate's mode is wrong at all          (where-wrong, Jaccard-like)
  likelihood     random questions / wrong answers; sum log P_m(answer)  (P(o | o != label) for errors)
"""
import sys; sys.path.insert(0,'.')
from load import *
M=load(); ds2=ds.set_index('index'); full=[m for m in M if len(M[m])==3000]; idx=ds['index'].tolist()
y=ds2.loc[idx,'label'].to_numpy(); A=arr(M,full,idx); nm,n,_=A.shape; rng=np.random.default_rng(7)
R=A[:,:,:5]; C=np.stack([(R==k).sum(-1) for k in range(5)],-1); mo=C[:,:,:4].argmax(-1); W=mo!=y
L=np.log((C+0.5)/(C+0.5).sum(-1,keepdims=True))
Pe=np.exp(L).copy(); Pe[:,np.arange(n),y]=0; Le=np.log(Pe/Pe.sum(-1,keepdims=True)+1e-300)
def pick(s,i): b=np.flatnonzero(s==s.max()); return rng.choice(b)==i
ks=[3,5,10,20,50]; out={}
for k in ks:
    r=dict(mode_hamming=0,lik_random=0,cond_hamming=0,co_error=0,lik_errors=0); T=300*nm
    for _ in range(300):
        for i in range(nm):
            q=rng.choice(n,k,replace=False); a=A[i,q,rng.integers(5,10,k)]
            r['mode_hamming']+=pick((mo[:,q]==a).sum(1),i); r['lik_random']+=pick(L[:,q,a].sum(1),i)
            pw=A[i,:,5:]; wr=np.argwhere((pw!=y[:,None])&(pw<4)); s=wr[rng.choice(len(wr),k,replace=False)]; q=s[:,0]; a=pw[q,s[:,1]]
            r['cond_hamming']+=pick((mo[:,q]==a).sum(1),i); r['co_error']+=pick(W[:,q].sum(1),i); r['lik_errors']+=pick(Le[:,q,a].sum(1),i)
    out[k]={m:v/T for m,v in r.items()}
print(f'{nm} models, chance {1/nm:.3f}')
print(pd.DataFrame(out).T.round(3).to_string())

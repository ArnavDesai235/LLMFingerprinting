import sys;sys.path.insert(0,'.')
from load import *
from sklearn.metrics import roc_auc_score
A=np.load('A_full.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist()
nm,n,_=A.shape;rng=np.random.default_rng(2)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
P=(cnt(A[:,:,:5])+0.5)/(cnt(A[:,:,:5])+0.5).sum(-1,keepdims=True); L=np.log(P)
EL=(P*L).sum(-1); VL=(P*L**2).sum(-1)-EL**2       # per model,row expected loglik & var under itself
# empirical calibration from the model's OWN runs 5-7 (genuine) on calibration rows; test on runs 8-9
for k in [10,20,50]:
    for method in ['naive','z']:
        gen=[];imp=[]
        for _ in range(300):
            for u in range(nm):
                rows=rng.choice(n,k,replace=False)
                for target,cands in [('in',range(nm)),('out',[c for c in range(nm) if c!=u])]:
                    pr=A[u,rows,rng.integers(8,10,k)]
                    ll=L[list(cands)][:,np.arange(k),pr] if False else L[list(cands)][:,rows,pr]
                    if method=='naive': s=ll.sum(1)/k
                    else: s=(ll.sum(1)-EL[list(cands)][:,rows].sum(1))/np.sqrt(VL[list(cands)][:,rows].sum(1))
                    (gen if target=='in' else imp).append(s.max())
        gen,imp=np.array(gen),np.array(imp)
        auc=roc_auc_score(np.r_[np.ones(len(gen)),np.zeros(len(imp))],np.r_[gen,imp])
        th=np.quantile(gen,0.05); rej=(imp<th).mean()
        print(f'k={k:3d} {method:5s} AUROC {auc:.3f}  reject unknown @95% accept {rej:.2f}')

import sys;sys.path.insert(0,'.')
from load import *
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist();src=np.load('src.npy',allow_pickle=True)
nm,n,_=A.shape; rng=np.random.default_rng(0)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
def ident(A,ks,trials=400,ref=slice(0,5),probe=range(5,10),rowpool=None,select=None):
    L=np.log((cnt(A[:,:,ref])+0.5)/(cnt(A[:,:,ref])+0.5).sum(-1,keepdims=True))
    G=np.log((cnt(A[:,:,ref].reshape(nm,1,-1))[:,0]+0.5)); G=G-np.log(np.exp(G).sum(-1,keepdims=True))
    pool=np.arange(A.shape[1]) if rowpool is None else rowpool
    res=[];conf=None
    for k in ks:
        ok=okm=okf=0;tot=0;C=np.zeros((nm,nm),int)
        for _ in range(trials):
            for i in range(nm):
                rows=rng.choice(pool,k,replace=False) if select is None else select[:k]
                pr=A[i,rows,rng.choice(list(probe),k)]
                p=L[:,rows,pr].sum(1).argmax(); C[i,p]+=1
                ok+=p==i; okf+=fam(full[p])==fam(full[i]); okm+=G[:,pr].sum(1).argmax()==i; tot+=1
        res.append((k,ok/tot,okf/tot,okm/tot));
        if k==10: conf=C
    return pd.DataFrame(res,columns=['k','exact','family','marginal']).set_index('k'),conf,L
ks=[1,3,5,10,20,50]
r,C,L=ident(A,ks); print('== random questions, 12 models, chance',round(1/nm,3)); print(r.round(3))
for i,m in enumerate(full):
    row=C[i].copy();acc=row[i]/row.sum();row[i]=-1;j=row.argmax()
    print(f'  {m:16s} k=10 {acc:.2f}  confused->{full[j]} {C[i,j]/C[i].sum():.2f}')
# by subset
for s in ['wikihow','activitynet']:
    rr,_,_=ident(A,[5,10,20],rowpool=np.where(src==s)[0],trials=200); print(s);print(rr.round(3))
# only rows where at least one model errs vs rows all right
mo=cnt(A)[...,:4].argmax(-1); allright=(mo==y).all(0)
rr,_,_=ident(A,[5,10,20],rowpool=np.where(~allright)[0],trials=200); print('rows with >=1 error');print(rr.round(3))
rr,_,_=ident(A,[5,10,20],rowpool=np.where(allright)[0],trials=200); print('rows all-correct (mode)');print(rr.round(3))
# mutual-information question selection (using reference runs only) then evaluate on probes, held-out row split
half=rng.permutation(n); trainrows,testrows=half[:1500],half[1500:]
P=np.exp(L)                                            # nm x n x 5
pm=P.mean(0); H=lambda p:-(p*np.log(p)).sum(-1)
MI=H(pm)-H(P).mean(0)
sel=np.argsort(-MI)
for k in [5,10,20]:
    rr,_,_=ident(A,[k],select=sel,trials=50); print('MI top',k, rr.exact.iloc[0].round(3))
np.save('MI.npy',MI)

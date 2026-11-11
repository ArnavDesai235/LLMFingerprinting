import sys;sys.path.insert(0,'.')
from load import *
import itertools, numpy.linalg as la
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist();src=np.load('src.npy',allow_pickle=True)
nm,n,_=A.shape;rng=np.random.default_rng(5)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
# ---- Claim 1: decompose the error signal: WHERE a model errs (binary) vs WHICH wrong option
C=cnt(A[:,:,:5]); corr=np.take_along_axis(C[...,:4],y[None,:,None],-1)[...,0]
pc=(corr+0.5)/(5+1.0)                       # P(correct) per model,row
Lfull=np.log((C+0.5)/(C+0.5).sum(-1,keepdims=True))
print('== errors-only ID: binary "where" vs full "where+which" ==')
for k in [3,5,10]:
    okb=okf=okw=tot=0
    for _ in range(300):
        for i in range(nm):
            pr=A[i,:,5:]; wr=np.argwhere((pr!=y[:,None])&(pr<4)); s=wr[rng.choice(len(wr),k,replace=False)]
            rows=s[:,0]; ans=pr[rows,s[:,1]]
            okb+=np.log(1-pc[:,rows]).sum(1).argmax()==i            # only "this row was wrong"
            okf+=Lfull[:,rows,ans].sum(1).argmax()==i
            # "which": conditional on wrong, P(ans | wrong)
            Cw=C[:,rows].astype(float).copy(); Cw[:,np.arange(k),y[rows]]=0; Pw=(Cw[:,np.arange(k),ans]+0.5)/(Cw[:,:, :4].sum(-1)+1.5)
            okw+=np.log(Pw).sum(1).argmax()==i; tot+=1
    print(f'k={k:2d} where-only {okb/tot:.3f}  which-only {okw/tot:.3f}  both {okf/tot:.3f}')
# ---- Claim 1 robustness: cross-domain (profile from one source, probe on other) not possible per-row;
# instead: row-split generalisation of MI selection (select on half A rows using runs0-4, test on held-out models? -> leave-one-model-out MI)
P=np.exp(Lfull); H=lambda p:-(p*np.log(p)).sum(-1)
print('\n== MI selection with the target model held out of selection ==')
for k in [5,10,20]:
    ok=tot=0
    for i in range(nm):
        others=[j for j in range(nm) if j!=i]; Po=P[others]; MI=H(Po.mean(0))-H(Po).mean(0); sel=np.argsort(-MI)[:k]
        for _ in range(100):
            pr=A[i,sel,rng.integers(5,10,k)]; ok+=Lfull[:,sel,pr].sum(1).argmax()==i; tot+=1
    print(k,round(ok/tot,3))
# ---- Claim 2: family effect on WHICH wrong answer, with position-bias null and model-level permutation
mo=C[...,:4].argmax(-1); W=mo!=y
bias=np.stack([np.bincount(mo[i],minlength=4)/n for i in range(nm)])
pairs=list(itertools.combinations(range(nm),2)); fams=np.array([fam(m) for m in full])
def excess(i,j):
    b=W[i]&W[j]; r=np.where(b)[0]
    obs=(mo[i,r]==mo[j,r]).mean()
    # null: each picks a wrong option by own position bias renormalised over wrong options of that row
    e=[]
    for t in r:
        w=[o for o in range(4) if o!=y[t]]; pi=bias[i,w]/bias[i,w].sum(); pj=bias[j,w]/bias[j,w].sum(); e.append((pi*pj).sum())
    return obs, np.mean(e), b.sum()
rec=[]
for i,j in pairs:
    o,e,nb=excess(i,j); acc=1-W.mean(1)
    rec.append(dict(i=i,j=j,obs=o,null=e,exc=o-e,nb=nb,fam=fams[i]==fams[j],accmin=min(acc[i],acc[j]),accdiff=abs(acc[i]-acc[j]),
                    kappa=None))
R=pd.DataFrame(rec)
print('\n== same-wrong vs position-bias null ==  mean obs %.3f  null %.3f'%(R.obs.mean(),R.null.mean()))
print(R.groupby('fam')[['obs','null','exc']].mean().round(3))
X=np.column_stack([np.ones(len(R)),R.accmin,R.accdiff,R.fam.astype(float)])
def fit(yv,X): return la.lstsq(X,yv,rcond=None)[0]
b=fit(R.exc,X); 
# model-level permutation: shuffle family labels across MODELS (keeps family sizes), preserves pair dependence
perm=[]
for _ in range(5000):
    f=rng.permutation(fams); Xp=X.copy(); Xp[:,3]=[f[i]==f[j] for i,j in pairs]; perm.append(fit(R.exc,Xp)[3])
p=(np.abs(perm)>=abs(b[3])).mean()
print(f'excess_same_wrong = {b[0]:.3f} + {b[1]:.3f}*acc_min + {b[2]:.3f}*acc_diff + {b[3]:.3f}*same_fam   model-level perm p={p:.4f}')
# within each family: pairs list
print(R[R.fam].assign(a=lambda d:[full[i] for i in d.i],b=lambda d:[full[j] for j in d.j])[['a','b','obs','null','exc','nb']].round(3).to_string(index=False))
# accuracy-matched check: for each within-family pair, compare against cross-family pairs with similar acc_min (±0.05)
d=[]
for _,r in R[R.fam].iterrows():
    m=R[(~R.fam)&(abs(R.accmin-r.accmin)<0.05)&(abs(R.accdiff-r.accdiff)<0.08)]
    if len(m): d.append(r.exc-m.exc.mean())
print('accuracy-matched within-minus-across excess: mean %.3f over %d within pairs, %d positive'%(np.mean(d),len(d),sum(x>0 for x in d)))

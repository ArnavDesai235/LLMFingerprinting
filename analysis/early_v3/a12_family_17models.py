"""Family analysis on all complete v3 models: D_H decomposition (co-error x same-wrong share), position-bias null,
model-level permutation, accuracy matching, per-family breakdown, k-way overlap, identification."""
import sys,itertools;sys.path.insert(0,'.')
from load import *
import numpy.linalg as la
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist()
EXCL=set(sys.argv[1:])            # optional models to drop (sensitivity)
keep=[i for i,m in enumerate(full) if m not in EXCL]; A=A[keep]; full=[full[i] for i in keep]
nm,n,_=A.shape;rng=np.random.default_rng(13);fams=np.array([fam(m) for m in full])
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
mo=cnt(A)[...,:4].argmax(-1);W=mo!=y;acc=1-W.mean(1)
bias=np.stack([np.bincount(mo[i],minlength=4)/n for i in range(nm)])
pairs=list(itertools.combinations(range(nm),2))
print(f'{nm} models:',dict(pd.Series(fams).value_counts()))
rec=[]
for i,j in pairs:
    both=W[i]&W[j]; r=np.where(both)[0]
    share=(mo[i,r]==mo[j,r]).mean()
    null=np.mean([ (lambda w:(bias[i,w]/bias[i,w].sum()*bias[j,w]/bias[j,w].sum()).sum())([o for o in range(4) if o!=y[t]]) for t in r])
    co=both.sum()/np.sqrt(W[i].sum()*W[j].sum())            # symmetric co-error rate (geometric mean of the two conditionals)
    lift=both.sum()/(W[i].mean()*W[j].mean()*n)
    dh_ij=1-(both.sum()/W[j].sum())*share; dh_ji=1-(both.sum()/W[i].sum())*share
    rec.append(dict(a=full[i],b=full[j],fam=fams[i]==fams[j],accmin=min(acc[i],acc[j]),accdiff=abs(acc[i]-acc[j]),
        co=co,lift=lift,share=share,null=null,excess=share-null,DH=(dh_ij+dh_ji)/2,nboth=both.sum()))
R=pd.DataFrame(rec)
print('\n== within vs across family (means over pairs) ==')
print(R.groupby('fam')[['DH','co','lift','share','null','excess']].mean().round(3))
X0=np.column_stack([np.ones(len(R)),R.accmin,R.accdiff])
def famtest(col,nperm=5000):
    X=np.column_stack([X0,R.fam.astype(float)]); b=la.lstsq(X,R[col],rcond=None)[0]
    perm=[]
    for _ in range(nperm):
        f=rng.permutation(fams); Xp=X.copy(); Xp[:,3]=[f[i]==f[j] for i,j in pairs]; perm.append(la.lstsq(Xp,R[col],rcond=None)[0][3])
    return b[3],(np.abs(perm)>=abs(b[3])).mean()
print('\n== family coefficient after controlling for accuracy (min, diff); model-level permutation p ==')
for col in ['DH','co','lift','share','excess']:
    b,p=famtest(col); print(f'  {col:7s} same_fam coef {b:+.3f}   p={p:.4f}')
# accuracy-matched, per family
print('\n== per family: within-family excess same-wrong minus accuracy-matched cross-family pairs ==')
for f in sorted(set(fams)):
    sub=R[R.fam&(R.a.map(fam)==f)]; d=[]
    for _,r in sub.iterrows():
        m=R[(~R.fam)&(abs(R.accmin-r.accmin)<0.05)&(abs(R.accdiff-r.accdiff)<0.08)]
        if len(m): d.append(r.excess-m.excess.mean())
    if d: print(f'  {f:8s} {len(sub)} pairs  mean diff {np.mean(d):+.3f}  positive {sum(x>0 for x in d)}/{len(d)}')
print('\n== within-family pairs ==')
print(R[R.fam].sort_values(['a','b'])[['a','b','DH','co','share','null','excess','nboth']].round(3).to_string(index=False))
# k-way: all models of a family wrong; identical wrong ending vs position-bias null; vs accuracy-matched cross-family groups
print('\n== whole-family all-wrong overlap ==')
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]; k=len(idx)
    if k<2: continue
    allw=W[idx].all(0); r=np.where(allw)[0]; ident=(mo[idx][:,r]==mo[idx[0]][r]).all(0).mean()
    nullv=np.mean([np.prod(bias[idx][:, [o for o in range(4) if o!=y[t]]]/bias[idx][:, [o for o in range(4) if o!=y[t]]].sum(1,keepdims=True),0).sum() for t in r])
    ca=np.sort(acc[idx]); vals=[]
    for c in itertools.combinations(range(nm),k):
        c=np.array(c)
        if len(set(fams[c]))<k or np.abs(np.sort(acc[c])-ca).max()>=0.07: continue
        aw=W[c].all(0); vals.append((mo[c][:,aw]==mo[c[0]][aw]).all(0).mean())
        if len(vals)>=400: break
    print(f'  {f:8s} k={k}  all-wrong {allw.sum():4d}  identical {ident:.2f}  (bias null {nullv:.2f};  {len(vals)} acc-matched cross-family groups {np.mean(vals) if vals else float("nan"):.2f})')
if EXCL: sys.exit()
R.to_csv('pairs_family_17.csv',index=False)
# identification (likelihood) with all models
C=cnt(A[:,:,:5]);L=np.log((C+0.5)/(C+0.5).sum(-1,keepdims=True))
print(f'\n== identification, {nm} models (chance {1/nm:.3f}) ==')
for k in [5,10,20,50]:
    ok=okf=okw=tot=0
    for _ in range(300):
        for i in range(nm):
            rows=rng.choice(n,k,replace=False); pr=A[i,rows,rng.integers(5,10,k)]; p=L[:,rows,pr].sum(1).argmax()
            ok+=p==i; okf+=fams[p]==fams[i]
            pw=A[i,:,5:]; wr=np.argwhere((pw!=y[:,None])&(pw<4)); s=wr[rng.choice(len(wr),k,replace=False)]
            okw+=L[:,s[:,0],pw[s[:,0],s[:,1]]].sum(1).argmax()==i; tot+=1
    print(f'  k={k:2d}  random questions: exact {ok/tot:.3f} family {okf/tot:.3f}   |  k wrong answers only: exact {okw/tot:.3f}')

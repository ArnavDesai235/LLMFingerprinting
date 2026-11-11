import sys;sys.path.insert(0,'.')
from load import *
import itertools, numpy.linalg as la
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist()
nm,n,_=A.shape;rng=np.random.default_rng(1)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
L=np.log((cnt(A[:,:,:5])+0.5)/(cnt(A[:,:,:5])+0.5).sum(-1,keepdims=True))
# errors-only identification: probe = k WRONG answers sampled from runs 5-9
print('== identification from k wrong answers only (runs 5-9 probe) ==')
for k in [1,3,5,10,20]:
    ok=tot=0
    for _ in range(300):
        for i in range(nm):
            pr=A[i,:,5:]; wr=np.argwhere((pr!=y[:,None])&(pr<4))
            s=wr[rng.choice(len(wr),k,replace=False)]
            rows=s[:,0]; ans=pr[rows,s[:,1]]
            ok+=L[:,rows,ans].sum(1).argmax()==i; tot+=1
    print(k, round(ok/tot,3))
# pairwise error correlation using mode answers
mo=cnt(A)[...,:4].argmax(-1); W=mo!=y
rec=[]
for i,j in itertools.combinations(range(nm),2):
    a,b=full[i],full[j]
    pi,pj=W[i].mean(),W[j].mean(); both=(W[i]&W[j])
    obs=(W[i]==W[j]).mean(); exp=pi*pj+(1-pi)*(1-pj); kappa=(obs-exp)/(1-exp)
    samewrong=(mo[i]==mo[j])[both].mean()   # chance ~1/3
    agree=(mo[i]==mo[j]).mean()
    rec.append(dict(a=a,b=b,same_fam=fam(a)==fam(b),agree=agree,kappa=kappa,same_wrong=samewrong,nboth=both.sum(),
        logmin=np.log(min(SIZE[a],SIZE[b])),logratio=abs(np.log(SIZE[a]/SIZE[b])),accmin=1-max(pi,pj),accdiff=abs(pi-pj)))
R=pd.DataFrame(rec)
print('\nwithin vs across family (mean):');print(R.groupby('same_fam')[['agree','kappa','same_wrong']].mean().round(3))
for t in ['agree','kappa','same_wrong']:
    X=np.column_stack([np.ones(len(R)),R.logmin,R.logratio,R.same_fam.astype(float)])
    b,*_=la.lstsq(X,R[t],rcond=None); pred=X@b; r2=1-((R[t]-pred)**2).sum()/((R[t]-R[t].mean())**2).sum()
    # permutation p for same_fam coef
    perm=[]
    for _ in range(2000):
        Xp=X.copy(); Xp[:,3]=rng.permutation(Xp[:,3]); perm.append(la.lstsq(Xp,R[t],rcond=None)[0][3])
    p=(np.abs(perm)>=abs(b[3])).mean()
    print(f'{t:10s} = {b[0]:.3f} + {b[1]:.3f}*log(min size) + {b[2]:.3f}*|log ratio| + {b[3]:.3f}*same_fam  R2={r2:.2f}  perm p(fam)={p:.3f}')
print('\nTop same_wrong pairs:');print(R.sort_values('same_wrong',ascending=False).head(8)[['a','b','same_wrong','kappa','nboth']].round(3).to_string(index=False))
print('Lowest:');print(R.sort_values('same_wrong').head(5)[['a','b','same_wrong','kappa','nboth']].round(3).to_string(index=False))
print('\nTop kappa:');print(R.sort_values('kappa',ascending=False).head(6)[['a','b','kappa','same_wrong']].round(3).to_string(index=False))
# chance level same_wrong: given both wrong, each picks among 3 wrong options w/ own position bias
R.to_csv('pairs.csv',index=False)
# hard rows: everyone wrong, do they converge?
allw=W.all(0); print('\nrows all 12 wrong:',allw.sum())
if allw.sum():
    from collections import Counter
    top=[Counter(mo[:,r]).most_common(1)[0][1] for r in np.where(allw)[0]]
    print('mean # models agreeing on the most popular wrong answer (of 12):',np.mean(top).round(2))
# per-model run-to-run consistency on errors: if wrong in mode, fraction of runs giving same wrong
for i,m in enumerate(full):
    rws=W[i]; c=cnt(A[i])[rws]; print(f'  {m:16s} err rows {rws.sum():4d}  mode-wrong share {(c.max(1)/10).mean():.2f}')

"""Report-style metrics on v3: asymmetric conditional Hamming (D_H), error-set Jaccard (D_J), and combinations.
Fingerprint F_a = per-row mode of model a's runs 0-4. Test runs R_b = each single run 5-9 of each model (12 x 5 = 60)."""
import sys;sys.path.insert(0,'.')
from load import *
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist()
nm,n,_=A.shape;rng=np.random.default_rng(7)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
F=cnt(A[:,:,:5])[...,:4].argmax(-1)            # fingerprints nm x n
T=[(b,A[b,:,r]) for b in range(nm) for r in range(5,10)]   # test runs
fams=np.array([fam(m) for m in full])
def dists(rows):
    """returns dict metric -> (ntest x nm) distance matrix on the given rows"""
    out={k:np.zeros((len(T),nm)) for k in ['H','J','Hsym','Hall','HJ','Hcorr']}
    Fe=F[:,rows]!=y[rows]
    for t,(b,R) in enumerate(T):
        Rr=R[rows]; Eb=(Rr!=y[rows])&(Rr<4)
        for a in range(nm):
            Fa=F[a,rows]; Ea=Fe[a]
            out['H'][t,a]=(Fa[Eb]!=Rr[Eb]).mean()                       # report Eq.3
            inter=(Ea&Eb).sum(); uni=(Ea|Eb).sum(); out['J'][t,a]=1-inter/max(uni,1)   # Eq.4
            both=Ea&Eb; out['Hsym'][t,a]=(Fa[both]!=Rr[both]).mean() if both.any() else 1  # same-wrong share on both-wrong
            out['Hall'][t,a]=(Fa!=Rr).mean()                            # plain Hamming on all rows
            out['Hcorr'][t,a]=(Fa[~Eb]!=Rr[~Eb]).mean()                 # disagreement on rows test run got right
    out['HJ']=0.5*out['H']+0.5*out['J']
    out['H+Hcorr']=out['H']+out['Hcorr']
    return out
truth=np.array([b for b,_ in T])
def score(D):
    o=np.argsort(D,1); top1=(o[:,0]==truth).mean(); fam1=(fams[o[:,0]]==fams[truth]).mean(); top3=(o[:,:3]==truth[:,None]).any(1).mean()
    return top1,fam1,top3
print('== closed set, all 3000 rows (60 test runs, 12 fingerprints, chance 8.3%) ==')
Dall=dists(np.arange(n))
for k,D in Dall.items(): print(f'  {k:8s} top1 {score(D)[0]:.3f}  family {score(D)[1]:.3f}  top3 {score(D)[2]:.3f}')
# nearest-neighbour confusions for D_H
o=Dall['H'].argmin(1); 
conf=pd.crosstab(pd.Series([full[t] for t in truth],name='true'),pd.Series([full[p] for p in o],name='pred'))
print('\nD_H top-1 misses:'); 
for t,p in zip(truth,o):
    pass
miss=pd.Series([(full[t],full[p]) for t,p in zip(truth,o) if t!=p]).value_counts(); print(miss.to_string())
# row-count sweeps: random order (20 bootstraps) vs variance-sorted (variance computed on fingerprint modes only)
var=np.std(F,0); vord=np.argsort(-var,kind='stable')
print('\n== row sweep: top1 for D_H / D_J (random order mean over 20 draws | variance-sorted) ==')
for m in [25,50,100,200,300,500,1000,2000,3000]:
    rs=[];rj=[]
    for _ in range(20 if m<3000 else 1):
        rows=rng.choice(n,m,replace=False); D=dists(rows); rs.append(score(D['H'])[0]); rj.append(score(D['J'])[0])
    Dv=dists(vord[:m])
    print(f'  rows {m:4d}  H rand {np.mean(rs):.3f}±{np.std(rs):.3f}  H var {score(Dv["H"])[0]:.3f}   J rand {np.mean(rj):.3f}  J var {score(Dv["J"])[0]:.3f}  Htop3 rand {score(D["H"])[2]:.3f}')
# open set: leave-one-model-out with threshold tau on D_H (report Eq. 6)
print('\n== open set on D_H: enrolled = 11 models, unseen = 1 (rotated); top-1 counts "other" as correct for unseen ==')
D=Dall['H']
for tau in [0.3,0.35,0.4,0.45,0.5,0.55,0.6]:
    acc=[];fa=[];fr=[]
    for u in range(nm):
        enr=[a for a in range(nm) if a!=u]; Dm=D[:,enr]; nn=np.array(enr)[Dm.argmin(1)]; dmin=Dm.min(1)
        pred=np.where(dmin<tau,nn,-1); tgt=np.where(truth==u,-1,truth)
        acc.append((pred==tgt).mean()); fr.append((pred[truth==u]==-1).mean()); fa.append((pred[truth!=u]==-1).mean())
    print(f'  tau {tau:.2f}  top1 {np.mean(acc):.3f}  unseen rejected {np.mean(fr):.3f}  enrolled wrongly rejected {np.mean(fa):.3f}')
gen=D[np.arange(len(T)),truth]; imp=np.array([np.min(np.delete(D[t],truth[t])) for t in range(len(T))])
print(f'  D_H genuine mean {gen.mean():.3f} (max {gen.max():.3f})  nearest impostor mean {imp.mean():.3f} (min {imp.min():.3f})')
np.save('DH.npy',D); np.save('DJ.npy',Dall['J'])

"""(1) report metrics on all 17 models over the 851 shared ActivityNet rows; (2) Mistral-note overlap stats per family."""
import sys,itertools;sys.path.insert(0,'.')
from load import *
M=load(); ms=list(M); common=sorted(set.intersection(*[set(M[m]) for m in ms])); y=ds.set_index('index').loc[common,'label'].to_numpy()
A=arr(M,ms,common); nm,n,_=A.shape; fams=np.array([fam(m) for m in ms]); rng=np.random.default_rng(8)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
F=cnt(A[:,:,:5])[...,:4].argmax(-1); T=[(b,A[b,:,r]) for b in range(nm) for r in range(5,10)]; truth=np.array([b for b,_ in T])
def D(rows):
    H=np.zeros((len(T),nm));J=np.zeros_like(H)
    Fe=F[:,rows]!=y[rows]
    for t,(b,R) in enumerate(T):
        Rr=R[rows];Eb=(Rr!=y[rows])&(Rr<4)
        H[t]=[(F[a,rows][Eb]!=Rr[Eb]).mean() if Eb.any() else 1 for a in range(nm)]
        J[t]=[1-(Fe[a]&Eb).sum()/max((Fe[a]|Eb).sum(),1) for a in range(nm)]
    return H,J
def sc(Dm): o=np.argsort(Dm,1); return (o[:,0]==truth).mean(),(fams[o[:,0]]==fams[truth]).mean(),(o[:,:3]==truth[:,None]).any(1).mean()
print(f'== 17 models, {n} ActivityNet rows, 85 test runs ==')
for m in [50,100,200,400,851]:
    r=[];
    for _ in range(20 if m<n else 1):
        H,J=D(rng.choice(n,m,replace=False)); r.append(sc(H)+sc(J))
    r=np.mean(r,0); print(f'  rows {m:3d}  D_H top1 {r[0]:.3f} fam {r[1]:.3f} top3 {r[2]:.3f} | D_J top1 {r[3]:.3f} fam {r[4]:.3f} top3 {r[5]:.3f}')
H,J=D(np.arange(n)); o=H.argmin(1)
print('  D_H misses at 851 rows:',pd.Series([f'{ms[t]}->{ms[p]}' for t,p in zip(truth,o) if t!=p]).value_counts().to_dict())
# ---------------- Mistral-note style overlap, 12 complete models, 3000 rows, mode over all 10 runs
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist();nm,n,_=A.shape
mo=cnt(A)[...,:4].argmax(-1);W=mo!=y;fams=np.array([fam(m) for m in full])
rec=[]
for i,j in itertools.combinations(range(nm),2):
    both=W[i]&W[j]; nb=both.sum(); same=(mo[i]==mo[j])[both].sum()
    lift=nb/(W[i].mean()*W[j].mean()*n)
    rec.append(dict(pair=f'{full[i]}–{full[j]}',fam=fams[i]==fams[j],both_wrong=nb,same_wrong=same,share=same/nb,lift=lift,
                    p_j_given_i=nb/W[i].sum(),p_i_given_j=nb/W[j].sum(),jaccard=nb/(W[i]|W[j]).sum()))
R=pd.DataFrame(rec)
print('\n== pairwise overlap (mode over 10 runs) ==')
print(R.groupby('fam')[['share','lift','jaccard','p_j_given_i']].mean().round(3))
print(R[R.fam].round(3).to_string(index=False))
# family-level k-way: all k wrong, all identical wrong ending; vs independence
print('\n== within-family k-way overlap (Mistral-note "all three wrong") ==')
for f in sorted(set(fams)):
    idx=np.where(fams==f)[0]
    if len(idx)<3: continue
    for combo in [idx]+[np.array(c) for c in itertools.combinations(idx,3)] if len(idx)>3 else [idx]:
        allw=W[combo].all(0); k=len(combo); ident=(mo[combo][:,allw]==mo[combo[0]][allw]).all(0).sum()
        exp_allw=np.prod(W[combo].mean(1))*n
        print(f'  {"+".join(full[c] for c in combo):45s} all-wrong {allw.sum():4d} ({allw.mean():.3f}; indep exp {exp_allw:.0f}, lift {allw.sum()/exp_allw:.2f})  identical {ident} ({ident/max(allw.sum(),1):.2f}; chance {(1/3)**(k-1):.2f})')
# accuracy-matched cross-family triples for comparison
print('\n  cross-family triples matched to each within-family triple by mean accuracy:')
acc=1-W.mean(1)
for f in ['falcon','phi','qwen']:
    idx=np.where(fams==f)[0]
    for combo in itertools.combinations(idx,3):
        combo=np.array(combo); allw=W[combo].all(0); wi=(mo[combo][:,allw]==mo[combo[0]][allw]).all(0).mean()
        ca=np.sort(acc[combo]); vals=[]
        for c in itertools.combinations(range(nm),3):
            c=np.array(c)
            if len(set(fams[c]))<3: continue
            if np.abs(np.sort(acc[c])-ca).max()<0.06:
                aw=W[c].all(0); vals.append((mo[c][:,aw]==mo[c[0]][aw]).all(0).mean())
        print(f'    {"+".join(full[c] for c in combo):45s} identical-when-all-wrong {wi:.2f}  vs {len(vals)} matched cross-family triples {np.mean(vals) if vals else float("nan"):.2f}')
R.to_csv('pairs_overlap.csv',index=False)

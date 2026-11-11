import sys;sys.path.insert(0,'.')
from load import *
A=np.load('A_full.npy');y=np.load('y.npy');full=pd.read_csv('full.csv').iloc[:,0].tolist()
nm,n,_=A.shape;rng=np.random.default_rng(12)
def cnt(a): return np.stack([(a==k).sum(-1) for k in range(5)],-1)
C=cnt(A[:,:,:5]);F=C[...,:4].argmax(-1);L=np.log((C+0.5)/(C+0.5).sum(-1,keepdims=True))
print('k wrong probe answers: conditional Hamming (mode) vs likelihood on the same wrong answers')
for k in [3,5,10,20]:
    h=l=t=0
    for _ in range(400):
        for i in range(nm):
            pr=A[i,:,5:]; wr=np.argwhere((pr!=y[:,None])&(pr<4)); s=wr[rng.choice(len(wr),k,replace=False)]
            rows=s[:,0]; ans=pr[rows,s[:,1]]
            h+=(-(F[:,rows]!=ans).sum(1)+rng.random(nm)*1e-3).argmax()==i; l+=L[:,rows,ans].sum(1).argmax()==i; t+=1
    print(f'  k={k:2d}  cond. Hamming {h/t:.3f}   likelihood {l/t:.3f}')
# decomposition check on mode fingerprints: D_H(a,b) = 1 - P(a wrong | b wrong) * P(same | both wrong)
W=F!=y; err=[]
for a in range(nm):
    for b in range(nm):
        if a==b: continue
        Eb=W[b]; DH=(F[a][Eb]!=F[b][Eb]).mean(); co=(W[a]&Eb).sum()/Eb.sum(); both=W[a]&Eb; sh=(F[a][both]==F[b][both]).mean()
        err.append(abs(DH-(1-co*sh)))
print('max |D_H - (1 - coerror*sameshare)| =',max(err))

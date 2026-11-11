import sys;sys.path.insert(0,'.')
from load import *
M=load(); ds2=ds.set_index('index')
full=[m for m in M if len(M[m])==3000]
idx=ds['index'].tolist(); y=ds2.loc[idx,'label'].to_numpy(); src=ds2.loc[idx,'source_id'].str.split('~').str[0].to_numpy()
A=arr(M,full,idx)   # models x rows x 10
pd.set_option('display.width',250)
rows=[]
for i,m in enumerate(full):
    a=A[i]; c=np.stack([(a==k).sum(1) for k in range(5)],1); mo=c[:,:4].argmax(1)
    ans=a[a<4]
    rows.append(dict(model=m,run_acc=(a==y[:,None]).mean(),mode_acc=(mo==y).mean(),
      acc_wiki=(mo==y)[src=='wikihow'].mean(),acc_anet=(mo==y)[src=='activitynet'].mean(),
      unanimous=(c.max(1)==10).mean(),mean_maxshare=(c.max(1)/10).mean(),
      **{f'p{k}':(ans==k).mean() for k in range(4)}))
P=pd.DataFrame(rows).set_index('model').sort_values('mode_acc',ascending=False)
print(P.round(3).to_string())
# oracle: fraction of rows any model gets right / none
mo=np.array([np.stack([(A[i]==k).sum(1) for k in range(4)],1).argmax(1) for i in range(len(full))])
corr=(mo==y)
print('rows all 12 right',corr.all(0).mean().round(3),' none right',(~corr).any(0).mean().round(3) if False else (~corr).all(0).mean().round(3))
np.save('A_full.npy',A); np.save('y.npy',y); pd.Series(full).to_csv('full.csv',index=False); np.save('src.npy',src)

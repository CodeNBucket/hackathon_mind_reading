"""How the number of PCA components (K) changes accuracy on an unseen person.

Same pipeline as end_to_end_example.py, repeated for many K, tested leave-one-subject-out on all 4 subjects.
Run: python pca_sweep.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
X = np.load(HERE/"eeg_epochs.npy"); trials = pd.read_csv(HERE/"trials.csv")
videos = pd.read_csv(HERE/"videos.csv").set_index("video_id"); V = np.load(HERE/"video_embeddings_toy.npy")
ch = json.load(open(HERE/"info.json"))["ch_names"]; times = np.arange(X.shape[2])/100-0.2; vid_ids=list(videos.index)
X = X - X[:,:,times<0].mean(2,keepdims=True)
post=[ch.index(c) for c in ch if c.startswith(("O","PO","P"))]; win=(times>=0)&(times<2.5)
F = X[:,post][:,:,win].reshape(len(X),len(post),-1,5).mean(-1).reshape(len(X),-1)
for s in trials.subject.unique():
    m=trials.subject.values==s; F[m]=(F[m]-F[m].mean(0))/F[m].std(0)
Vz=(V-V.mean(0))/V.std(0)
def cos(a,b): return a@b.T/np.linalg.norm(a,axis=-1,keepdims=True)/np.linalg.norm(b,axis=-1)
res={}
for K in [1,2,5,10,20,40,80,150,300,650]:
    t1=[];t5=[];var=[]
    for test_s in trials.subject.unique():
        trm=trials.subject.values!=test_s
        A_te=np.stack([F[(trials.subject.values==test_s)&(trials.video_id.values==v)].mean(0) for v in vid_ids])
        Atr=F[trm]; ytr=np.array([vid_ids.index(v) for v in trials.video_id.values[trm]])
        mu=Atr.mean(0); _,S_,Vt=np.linalg.svd(Atr-mu,full_matrices=False)
        k=min(K,Vt.shape[0]); Ztr=(Atr-mu)@Vt[:k].T; Zte=(A_te-mu)@Vt[:k].T
        W=np.linalg.solve(Ztr.T@Ztr+10*np.eye(k),Ztr.T@Vz[ytr])
        sims=cos(Zte@W,Vz); t=np.arange(20)
        t1.append((sims.argmax(1)==t).mean()); t5.append(np.mean([t[r] in np.argsort(-sims[r])[:5] for r in range(20)]))
        var.append((S_[:k]**2).sum()/(S_**2).sum())
    print(f"K={K:4d}  var kept {100*np.mean(var):5.1f}%   top-1 {100*np.mean(t1):5.1f}%   top-5 {100*np.mean(t5):5.1f}%")

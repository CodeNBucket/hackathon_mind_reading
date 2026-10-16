"""End-to-end toy pipeline on the synthetic data: EEG -> PCA -> map to video vector -> pick closest video.

Run: python end_to_end_example.py   (numpy + pandas only)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

np.set_printoptions(precision=2, suppress=True)
HERE = Path(__file__).parent
X = np.load(HERE / "eeg_epochs.npy")                 # (400 trials, 33 electrodes, 320 time points)
trials = pd.read_csv(HERE / "trials.csv")
videos = pd.read_csv(HERE / "videos.csv").set_index("video_id")
V = np.load(HERE / "video_embeddings_toy.npy")       # (20 videos, 16): stand-in for a video model's output
ch = json.load(open(HERE / "info.json"))["ch_names"]
times = np.arange(X.shape[2]) / 100 - 0.2
vid_ids = list(videos.index)

# ---- STEP 1: EEG -> a feature row per trial ------------------------------------------------
X = X - X[:, :, times < 0].mean(axis=2, keepdims=True)              # baseline correction
post = [ch.index(c) for c in ch if c.startswith(("O", "PO", "P"))]   # back-of-head electrodes
win = (times >= 0) & (times < 2.5)
F = X[:, post][:, :, win].reshape(len(X), len(post), -1, 5).mean(-1).reshape(len(X), -1)  # 50 ms bins
for s in trials.subject.unique():
    m = trials.subject.values == s
    F[m] = (F[m] - F[m].mean(0)) / F[m].std(0)
print(f"STEP 1  each trial = {F.shape[1]} numbers ({len(post)} electrodes x {F.shape[1] // len(post)} time bins)")

test_s = "sub-04"
# Training rows: every single trial of the 3 training people
tr_rows = trials.subject.values != test_s
# Test rows: the new person's 5 repetitions of each video averaged (noise removal)
te_keys = [(test_s, v) for v in vid_ids]
A_te = np.stack([F[(trials.subject.values == s) & (trials.video_id.values == v)].mean(0) for s, v in te_keys])
A = np.vstack([F[tr_rows], A_te])
vid = np.concatenate([[vid_ids.index(v) for v in trials.video_id.values[tr_rows]], np.arange(len(vid_ids))])
tr = np.arange(len(A)) < tr_rows.sum(); te = ~tr

# ---- STEP 2: PCA (fit on training people only) ---------------------------------------------
K = 40
mu = A[tr].mean(0)
_, S, Vt = np.linalg.svd(A[tr] - mu, full_matrices=False)
Z = (A - mu) @ Vt[:K].T                                              # PC scores: K numbers per row
print(f"STEP 2  PCA: {F.shape[1]} numbers -> {K} numbers per trial; "
      f"they keep {100 * (S[:K]**2).sum() / (S**2).sum():.0f}% of the variation")
print(f"        {test_s}, first 4 PC scores for 3 videos:")
for v in vid_ids[:3]:
    r = np.flatnonzero(te & (vid == vid_ids.index(v)))[0]
    print(f"        {v} ({videos.category[v]:6s}) PC1..PC4 = {Z[r, :4]}")

# ---- STEP 3: video -> vector (a video model would do this; here a toy 8-number vector) ------
Vz = (V - V.mean(0)) / V.std(0)
print(f"STEP 3  video vectors (first 3 videos, {V.shape[1]} numbers each):")
for k in range(3):
    print(f"        {vid_ids[k]} = {Vz[k]}")

# ---- STEP 4: learn the bridge EEG PCs -> video vector (ridge regression) --------------------
lam = 10.0
Ztr, Ytr = Z[tr], Vz[vid[tr]]
W = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(K), Ztr.T @ Ytr)     # (K x video-vector length) weights
print(f"STEP 4  learned a {K} x {V.shape[1]} weight matrix on 3 training people (the only trained part)")

# ---- STEP 5: new person's trial -> predicted video vector -> closest candidate --------------
def cos(a, b):
    return a @ b.T / np.linalg.norm(a, axis=-1, keepdims=True) / np.linalg.norm(b, axis=-1)
pred = Z[te] @ W
sims = cos(pred, Vz)                                                 # (20 test rows x 20 candidates)
hits = np.flatnonzero(sims.argmax(1) == vid[te])
i = int(hits[0]) if len(hits) else 0          # show a correctly matched example
true_v = vid_ids[vid[te][i]]
order = np.argsort(-sims[i])[:5]
print(f"STEP 5  {test_s} watched {true_v}; similarity to the top 5 candidates:")
for j in order:
    print(f"        {vid_ids[j]} ({videos.category[vid_ids[j]]:6s}) cos = {sims[i, j]:+.2f}" + ("   <- correct" if vid_ids[j] == true_v else ""))
top1 = (sims.argmax(1) == vid[te]).mean()
top5 = np.mean([vid[te][r] in np.argsort(-sims[r])[:5] for r in range(len(sims))])
print(f"RESULT  {test_s} (unseen person): top-1 {top1:.0%}, top-5 {top5:.0%}  (chance: 5% and 25%)")

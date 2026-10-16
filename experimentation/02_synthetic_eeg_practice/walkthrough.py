"""Walk through the synthetic dataset the way you would on day one of the hackathon.

Run:  python walkthrough.py   (needs numpy, pandas, matplotlib)
Produces three PNGs and prints a tiny leave-one-subject-out decoding result.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
X = np.load(HERE / "eeg_epochs.npy")          # (trials, channels, time) in microvolts
trials = pd.read_csv(HERE / "trials.csv")     # one row per trial: subject, video_id, repetition...
videos = pd.read_csv(HERE / "videos.csv")     # one row per video: category, luminance, motion...
info = json.load(open(HERE / "info.json"))
ch = info["ch_names"]
times = np.arange(X.shape[2]) / info["sfreq_hz"] + info["tmin_s"]

# 1) Audit: always print shapes and counts first --------------------------------------
print("EEG array:", X.shape, "= trials x channels x time")
print("Subjects:", trials.subject.nunique(), "| videos:", trials.video_id.nunique(),
      "| reps per video per subject:", trials.groupby(["subject", "video_id"]).size().unique())

# 2) One trial vs the average of many: why averaging matters ----------------------------
oz = ch.index("Oz")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(times, X[0, oz], color="0.7", lw=0.8, label="one trial (Oz)")
ax.plot(times, X[:, oz].mean(0), color="C0", lw=2, label=f"average of {len(X)} trials (Oz)")
ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-0.2, 1.0)
ax.set_xlabel("time after video onset (s)"); ax.set_ylabel("voltage (µV)")
ax.set_title("A single trial is mostly noise; averaging reveals the brain response (ERP)")
ax.legend(); fig.tight_layout(); fig.savefig(HERE / "fig1_single_vs_average.png", dpi=130)

# 3) Faces vs everything else at a face-sensitive channel (the N170) ---------------------
cat = trials.video_id.map(videos.set_index("video_id").category)
p8 = ch.index("P8")
fig, ax = plt.subplots(figsize=(8, 3.5))
for c, col in [("face", "C3"), ("animal", "C1"), ("scene", "C2"), ("object", "C4")]:
    ax.plot(times, X[cat.values == c, p8].mean(0), color=col, label=c)
ax.axvline(0, color="k", lw=0.8); ax.axvspan(0.13, 0.20, color="0.9"); ax.set_xlim(-0.2, 1.0)
ax.set_xlabel("time after video onset (s)"); ax.set_ylabel("voltage (µV)")
ax.set_title("Faces give a deeper dip ~170 ms at P8 (the N170)")
ax.legend(); fig.tight_layout(); fig.savefig(HERE / "fig2_n170_by_category.png", dpi=130)

# 4) An artifact: a trial with an eye blink on the frontal channel ----------------------
i = int(np.flatnonzero(trials.has_blink.values)[0])
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(times, X[i, ch.index("Fp1")], label="Fp1 (above the eye)")
ax.plot(times, X[i, oz], label="Oz (visual cortex)")
ax.set_xlabel("time after video onset (s)"); ax.set_ylabel("voltage (µV)")
ax.set_title("An eye blink: huge on frontal channels, barely visible at the back")
ax.legend(); fig.tight_layout(); fig.savefig(HERE / "fig3_blink_artifact.png", dpi=130)

# 5) Tiny decoder with leave-one-subject-out ----------------------------------------------
X = X - X[:, :, times < 0].mean(axis=2, keepdims=True)                   # baseline correction
post = [ch.index(c) for c in ch if c.startswith(("O", "PO", "P"))]       # posterior channels only
win = (times >= 0.0) & (times < 2.5)                                       # first 2.5 s after onset
binned = X[:, post][:, :, win].reshape(len(X), len(post), -1, 5).mean(-1)  # average into 50 ms bins
feat = binned.reshape(len(X), -1)
for s in trials.subject.unique():                                          # z-score within each subject
    m = trials.subject.values == s
    feat[m] = (feat[m] - feat[m].mean(0)) / feat[m].std(0)
vids = sorted(trials.video_id.unique())

def loso(average_reps):
    accs = []
    for test_s in trials.subject.unique():
        tr, te = trials.subject.values != test_s, trials.subject.values == test_s
        centroids = np.stack([feat[tr & (trials.video_id.values == v)].mean(0) for v in vids])
        if average_reps:   # average the 5 repetitions of each video for the test subject
            Xte = np.stack([feat[te & (trials.video_id.values == v)].mean(0) for v in vids]); yte = np.arange(len(vids))
        else:
            Xte = feat[te]; yte = np.array([vids.index(v) for v in trials.video_id.values[te]])
        r = np.corrcoef(Xte, centroids)[: len(Xte), len(Xte):]           # correlation to each video's template
        accs.append((r.argmax(1) == yte).mean())
    return np.mean(accs)

print(f"Chance: {1 / len(vids):.1%}")
print(f"LOSO top-1, single trials:      {loso(False):.1%}")
print(f"LOSO top-1, 5 reps averaged:    {loso(True):.1%}")

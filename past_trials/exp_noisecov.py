"""Which covariance should the channel whitening use?

    python past_trials\\exp_noisecov.py                      # all three modes, 3 splits
    python past_trials\\exp_noisecov.py --modes noise --seeds 0

Modes (shrinkage fixed at 0.3, everything else = final model):
  file   current method: covariance of the whole training file (all videos, signal + noise).
         Should reproduce validate() of the final model (~20.9% top-5).
  train  same, but estimated only from the videos the model is fitted on (hidden videos excluded).
         Control: separates "less data" from "noise instead of signal+noise".
  noise  covariance of the NOISE only: for each fitting video, this subject's EEG minus the mean
         EEG of the other subjects for the same video (the shared, video-driven response is removed).
         This is the textbook multivariate noise normalisation (Guggenmos et al., 2018).

Why it might help: the 'file' covariance contains the video response itself, so whitening partly
flattens the signal we want to match. The noise covariance only flattens what is NOT video-driven.

Rules: 'train' and 'noise' use only the fitting videos (they need which-video labels), so the hidden
videos stay unseen, exactly as the real test files would be. Scored with the same held-out
training-video protocol as mindreading.model.validate (15% hidden, 3 splits). No test data used.

Results are merged into results/noisecov.json (one entry per mode x split), and the summary
prints the modes whose 3 splits are complete.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mindreading import model as M  # noqa: E402
from mindreading.config import CACHE_DIR, RESULTS_DIR, SUBJECTS, check_data_dir  # noqa: E402
from mindreading.data import load_train  # noqa: E402
from mindreading.preprocess import downsample, normalise, window  # noqa: E402

SHRINK = 0.3


# ---------------------------------------------------------------- EEG at 100 Hz, cached

def load_downsampled(path):
    """{subject: (video ids, EEG (n, 17, 300) at 100 Hz, not whitened)}; cached after the first run."""
    if path.exists():
        z = np.load(path)
        return {s: (z[f"ids{s}"], z[f"x{s}"]) for s in SUBJECTS}
    out = {}
    for s in SUBJECTS:
        ids, e = load_train(s)
        out[s] = (ids, downsample(e))
        print(f"  downsampled sub-{s:02d}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"ids{s}": out[s][0] for s in SUBJECTS}, **{f"x{s}": out[s][1] for s in SUBJECTS})
    return out


# ---------------------------------------------------------------- whitening helpers

def chan_cov(x):
    """17 x 17 channel covariance over all recordings and time points of x (n, 17, T)."""
    return np.cov(x.transpose(1, 0, 2).reshape(x.shape[1], -1))


def zca(C, shrink=SHRINK):
    """Same maths as preprocess.whiten_channels: shrink towards the identity, then C^(-1/2)."""
    C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    return (V / np.sqrt(np.maximum(w, 1e-8))) @ V.T


def noise_covs(ds, fit_ids):
    """{subject: covariance of (this subject - mean of the other subjects)} on the fitting videos.
    Each subject is z-scored per channel first so subjects with different amplitudes mix fairly;
    each result is converted back to that subject's own units. Accumulated video by video
    (no big stacked arrays), so it fits in little memory."""
    stats = {s: (x.mean((0, 2), keepdims=True)[0], x.std((0, 2), keepdims=True)[0]) for s, (_, x) in ds.items()}
    row = {s: {int(v): i for i, v in enumerate(ids)} for s, (ids, _) in ds.items()}

    def z(s, v):
        mu, sd = stats[s]
        return (ds[s][1][row[s][v]] - mu) / sd                      # (17, T)

    acc = {s: [np.zeros((17, 17)), np.zeros(17), 0] for s in ds}  # sum r r^T, sum r, n samples
    for v in (int(v) for v in fit_ids):
        have = [s for s in ds if v in row[s]]
        if len(have) < 2:
            continue
        zs = {s: z(s, v) for s in have}
        total = sum(zs.values())
        for s in have:
            r = zs[s] - (total - zs[s]) / (len(have) - 1)          # this subject minus the others' mean
            a = acc[s]
            a[0] += r @ r.T
            a[1] += r.sum(1)
            a[2] += r.shape[1]
    out = {}
    for s, (S, m, n) in acc.items():
        m = m / n
        C = (S - n * np.outer(m, m)) / (n - 1)
        D = np.diag(stats[s][1].ravel())
        out[s] = D @ C @ D
    return out


class CovModel(M.Model):
    """The final model, with the whitening matrix chosen by `mode` (see module docstring)."""

    def __init__(self, cfg, feats, mode, ds):
        super().__init__(cfg, feats)
        self.mode, self.ds = mode, ds

    def set_fit_videos(self, fit_ids):
        self.eeg, self.W = {}, {}
        noise = noise_covs(self.ds, fit_ids) if self.mode == "noise" else None
        for s, (ids, x) in self.ds.items():
            if self.mode == "file":
                C = chan_cov(x)
            elif self.mode == "train":
                C = chan_cov(x[np.isin(ids, fit_ids)])
            elif self.mode == "noise":
                C = noise[s]
            else:
                raise ValueError(self.mode)
            self.W[s] = zca(C)

    def train_eeg(self, sub):
        if sub not in self.eeg:
            ids, x = self.ds[sub]
            xw = np.einsum("ij,njt->nit", self.W[sub], x).astype(np.float32)
            xw = window(normalise(xw, "set"), self.cfg.tmin, self.cfg.tmax)
            self.eeg[sub] = (ids, xw.reshape(len(xw), -1))
        return self.eeg[sub]


# ---------------------------------------------------------------- validation (same protocol as model.validate)

def run_split(mode, seed, cfg, feats, ds):
    fit_ids, val_ids = M.split_train_videos(seed=seed)
    model = CovModel(cfg, feats, mode, ds)
    model.set_fit_videos(fit_ids)
    model.fit_shared(fit_ids)
    per_sub = []
    for s in SUBJECTS:
        members = {**model.fit_subject(s, fit_ids), **model.shared}
        ids, Y = model.train_eeg(s)
        m = np.isin(ids, val_ids)
        cand = np.unique(ids[m])
        sc = model.scores(members, Y[m], cand)["ensemble"]
        per_sub.append(M.topk(sc, np.searchsorted(cand, ids[m])))
    return {q: float(np.mean([r[q] for r in per_sub])) for q in ("top1", "top5", "median_rank")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="file,train,noise")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ds-cache", type=Path, default=CACHE_DIR / "eeg_100hz_unwhitened.npz")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "noisecov.json")
    ap.add_argument("--blocks", default=None,
                    help="feature blocks to use, e.g. 'clip' for a quick run (default: the final model's four)")
    a = ap.parse_args()

    check_data_dir()
    cfg = M.Config(whiten=SHRINK)
    if a.blocks:
        cfg.blocks = {b: cfg.blocks[b] for b in a.blocks.split(",")}
    feats = M.VideoFeatures(cfg.blocks)
    ds = load_downsampled(a.ds_cache)

    done = json.loads(a.out.read_text()) if a.out.exists() else []
    for mode in a.modes.split(","):
        for seed in (int(s) for s in a.seeds.split(",")):
            t0 = time.time()
            r = run_split(mode, seed, cfg, feats, ds)
            done = [d for d in done if not (d["mode"] == mode and d["seed"] == seed)]
            done.append({"mode": mode, "seed": seed, **r})
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(done, indent=1))
            print(f"{mode:6s} split {seed}: top-1 {r['top1']:.1%}  top-5 {r['top5']:.1%}  "
                  f"median {r['median_rank']:.1f}  ({time.time() - t0:.0f} s)", flush=True)

    print("\nmode   | top-1 | top-5 | median rank   (complete modes, 3 splits averaged)")
    for mode in ("file", "train", "noise"):
        rows = [d for d in done if d["mode"] == mode]
        if len(rows) == 3:
            avg = {q: np.mean([d[q] for d in rows]) for q in ("top1", "top5", "median_rank")}
            print(f"{mode:6s} | {avg['top1']:5.1%} | {avg['top5']:5.1%} | {avg['median_rank']:.1f}")


if __name__ == "__main__":
    main()

"""Fast experiments on the EEG side (encoder direction), validated on held-out TRAINING videos.

    .venv\\Scripts\\python.exe scripts\\exp_eeg.py <experiment>
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mindreading.config import CACHE_DIR, SUBJECTS  # noqa: E402
from mindreading.data import load_test, load_train  # noqa: E402
from mindreading.pipeline import split_train_videos  # noqa: E402

ALPHAS = np.logspace(-3, 7, 21)
SEEDS = (0, 1, 2)  # validation splits averaged, to reduce noise


def raw_cache():
    """All subjects' raw EEG downsampled to 100 Hz (no normalisation), cached once."""
    p = CACHE_DIR / "eeg_raw100.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    out = {}
    for s in SUBJECTS:
        ids, e = load_train(s)
        out[f"trn_ids_{s}"] = ids
        out[f"trn_{s}"] = decimate(e, 5, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)
        t, lab = load_test(s)
        out[f"tst_{s}"] = decimate(t, 5, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)
        if lab is not None:
            out[f"tst_lab_{s}"] = lab
    np.savez(p, **out)
    return out


def whiten_channels(x, shrink=0.1):
    """Decorrelate electrodes with this set's own channel covariance (shrunk towards identity)."""
    flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    C = np.cov(flat)
    C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T
    return np.einsum("ij,njt->nit", W, x).astype(np.float32)


def prep(x, band=None, car=False, norm="set", tmin=0.0, tmax=1.0, smooth=0, whiten=0):
    if band is not None:
        lo, hi = band
        if lo:
            sos = butter(4, [lo, hi], btype="band", fs=100, output="sos")
        else:
            sos = butter(4, hi, btype="low", fs=100, output="sos")
        x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    if car:
        x = x - x.mean(axis=1, keepdims=True)
    if whiten:
        x = whiten_channels(x, whiten)
    if norm == "set":
        x = (x - x.mean(axis=(0, 2), keepdims=True)) / (x.std(axis=(0, 2), keepdims=True) + 1e-8)
    a, b = int(round(tmin * 100)), int(round(tmax * 100))
    x = x[:, :, a:b]
    if smooth > 1:
        k = np.ones(smooth) / smooth
        x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, x)
    return x.astype(np.float32)


def rowz(a):
    a = a - a.mean(1, keepdims=True)
    return a / (a.std(1, keepdims=True) + 1e-8)


def wcorr(real, pred, w=None):
    """Weighted Pearson correlation between every real row and every predicted row."""
    if w is not None:
        sw = np.sqrt(w / w.sum() * len(w))
        real, pred = real * sw, pred * sw
    return rowz(real) @ rowz(pred).T / real.shape[1]


PER_TARGET = False


def fit_encoder(V, Y):
    # centre only: feature blocks are already scaled, so their relative weights must survive
    m = [StandardScaler(with_std=False).fit(V)]
    r = RidgeCV(alphas=ALPHAS, alpha_per_target=PER_TARGET).fit(m[0].transform(V), Y)
    return lambda X: r.predict(m[0].transform(X))


def reliability(V, Y, folds=5):
    """Out-of-fold correlation between predicted and real EEG, per (channel, time) feature."""
    pred = np.zeros_like(Y)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(V):
        pred[te] = fit_encoder(V[tr], Y[tr])(V[te])
    pz = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
    yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    return (pz * yz).mean(0)


def topk(S, true_cols):
    ts = S[np.arange(len(S)), true_cols]
    r = (S > ts[:, None]).sum(1) + 1
    return np.array([(r <= 1).mean(), (r <= 5).mean()])


def evaluate(feats_fn, prep_kw, mode="subject", weight_pow=0, use_ga=False, tag="", verbose=True,
             split="val", seed=None):
    """feats_fn(ids) -> video feature matrix.
    mode: 'subject' (own data), 'pooled' (all subjects), 'both' (sum of z-scored scores).
    weight_pow: 0 = plain correlation; p > 0 = weight features by max(r, 0)**p (out-of-fold).
    use_ga: also train on the grand-average EEG across subjects (per video) and add its scores.
    split: 'val' = held-out training videos (averaged over SEEDS); 'test' = sub-01..04 test sets."""
    if split == "val" and seed is None:
        res = np.mean([evaluate(feats_fn, prep_kw, mode, weight_pow, use_ga, verbose=False, seed=s)
                       for s in SEEDS], axis=0)
        if verbose:
            print(f"{tag:45s} top1 {res[:, 0].mean():.3f} top5 {res[:, 1].mean():.3f}   per-subj top5 "
                  + " ".join(f"{v:.2f}" for v in res[:, 1]), flush=True)
        return res
    raw = raw_cache()
    fit_ids, val_ids = split_train_videos(seed=seed or 0)
    data = {}
    for s in SUBJECTS:
        x = prep(raw[f"trn_{s}"], **prep_kw)
        data[s] = (raw[f"trn_ids_{s}"], x.reshape(len(x), -1))
    if split == "test":
        fit_ids = np.arange(1, 1001)
    subs = SUBJECTS if split == "val" else [1, 2, 3, 4]

    def train(ids, Y):
        V = feats_fn(ids)
        f = fit_encoder(V, Y)
        w = np.clip(reliability(V, Y), 0, None) ** weight_pow if weight_pow else None
        return f, w

    models = {}
    if mode in ("pooled", "both"):
        I = np.concatenate([data[s][0][np.isin(data[s][0], fit_ids)] for s in SUBJECTS])
        Y = np.concatenate([data[s][1][np.isin(data[s][0], fit_ids)] for s in SUBJECTS])
        models["pooled"] = train(I, Y)
    if use_ga:
        ga_ids = np.array([v for v in fit_ids if any(v in data[s][0] for s in SUBJECTS)])
        Y = np.stack([np.mean([data[s][1][np.searchsorted(data[s][0], v)] for s in SUBJECTS
                               if v in set(data[s][0])], 0) for v in ga_ids])
        models["ga"] = train(ga_ids, Y)
    res = []
    for s in subs:
        ids, Y = data[s]
        own = {}
        if mode in ("subject", "both"):
            m = np.isin(ids, fit_ids)
            own["subject"] = train(ids[m], Y[m])
        if split == "val":
            m = np.isin(ids, val_ids)
            real, cand = Y[m], np.unique(ids[m])
            true_cols = np.searchsorted(cand, ids[m])
        else:
            x = prep(raw[f"tst_{s}"], **prep_kw)
            real, cand = x.reshape(len(x), -1), np.arange(1001, 1103)
            true_cols = raw[f"tst_lab_{s}"] - 1001
        S = 0
        for name, (f, w) in {**own, **models}.items():
            S = S + rowz(wcorr(real, f(feats_fn(cand)), w))
        res.append(topk(S, true_cols))
    res = np.array(res)
    if verbose:
        print(f"{tag:45s} top1 {res[:, 0].mean():.3f} top5 {res[:, 1].mean():.3f}   per-subj top5 "
              + " ".join(f"{v:.2f}" for v in res[:, 1]), flush=True)
    return res


def clip_feats(key="mean", name="clip_b32"):
    z = np.load(CACHE_DIR / f"{name}.npz")
    pos = {int(v): i for i, v in enumerate(z["ids"])}
    arr = z[key].reshape(len(z["ids"]), -1)
    return lambda ids: arr[[pos[int(v)] for v in ids]]


def block(arr_by_id, n_pca=None):
    """Standardise a feature block (fit on training videos only), optionally PCA, scale to unit total variance."""
    from sklearn.decomposition import PCA
    ids = np.array(sorted(arr_by_id))
    X = np.stack([arr_by_id[i] for i in ids]).astype(np.float32)
    tr = ids <= 1000
    sc = StandardScaler().fit(X[tr])
    Z = sc.transform(X)
    if n_pca and n_pca < Z.shape[1]:
        Z = PCA(n_pca, random_state=0).fit(Z[tr]).transform(Z)
        Z /= Z[tr].std(0, keepdims=True).mean()
    Z /= np.sqrt(Z.shape[1])
    return {int(i): z for i, z in zip(ids, Z)}


def load_block(name, key, select=None, n_pca=None):
    z = np.load(CACHE_DIR / f"{name}.npz")
    arr = z[key].astype(np.float32)
    if select is not None:
        arr = select(arr)
    arr = arr.reshape(len(arr), -1)
    return block({int(i): a for i, a in zip(z["ids"], arr)}, n_pca)


def concat(*blocks, weights=None):
    weights = weights or [1.0] * len(blocks)
    return lambda ids: np.concatenate([w * np.stack([b[int(i)] for i in ids]) for b, w in zip(blocks, weights)], 1)


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "eeg"
    F = clip_feats()
    base = dict(tmin=0.0, tmax=1.0)
    t0 = time.time()
    if exp == "eeg":
        evaluate(F, base, tag="baseline (subject, set-norm, 0-1s)")
        evaluate(F, {**base, "band": (0.1, 30)}, tag="band 0.1-30 Hz")
        evaluate(F, {**base, "band": (None, 15)}, tag="lowpass 15 Hz")
        evaluate(F, {**base, "band": (None, 8)}, tag="lowpass 8 Hz")
        evaluate(F, {**base, "car": True}, tag="common average reference")
        evaluate(F, {**base, "tmin": 0.05, "tmax": 0.8}, tag="window 0.05-0.8")
        evaluate(F, {**base, "tmax": 1.3}, tag="window 0-1.3")
        evaluate(F, base, mode="both", tag="subject + pooled")
        evaluate(F, base, use_ga=True, tag="subject + grand-average target")
        evaluate(F, base, weight_pow=1, tag="reliability-weighted correlation (p=1)")
        evaluate(F, base, weight_pow=2, tag="reliability-weighted correlation (p=2)")
    if exp == "eeg2":
        car = {**base, "car": True}
        evaluate(F, car, tag="CAR")
        for hi in (12, 8, 5, 3):
            evaluate(F, {**car, "band": (None, hi)}, tag=f"CAR + lowpass {hi} Hz")
        evaluate(F, {**car, "band": (0.5, 8)}, tag="CAR + band 0.5-8 Hz")
        evaluate(F, {**car, "tmax": 0.8}, tag="CAR, window 0-0.8")
        evaluate(F, {**car, "tmax": 1.5}, tag="CAR, window 0-1.5")
        evaluate(F, car, mode="both", tag="CAR + subject+pooled")
        evaluate(F, car, mode="both", use_ga=True, tag="CAR + subject+pooled+GA")
    best = {**base, "car": True}
    ens = dict(mode="both", use_ga=True)
    if exp == "eeg3":
        evaluate(F, {**best, "band": (0.5, 8)}, **ens, tag="CAR + 0.5-8 Hz + ensemble")
        evaluate(F, {**best, "band": (None, 12)}, **ens, tag="CAR + lowpass 12 + ensemble")
    if exp.startswith("feat"):
        prep_kw = {**best, **({"band": (None, 12)} if "lp" in exp else {})}
        avail = {p.stem for p in CACHE_DIR.glob("*.npz")}
        cands = {"clip_b32 mean": load_block("clip_b32", "mean")}
        cands["clip_b32 first-second frames"] = load_block("clip_b32", "frames", lambda a: a[:, :3])
        if "lowlevel" in avail:
            cands["lowlevel 0-1.2 s"] = load_block("lowlevel", "timecourse", lambda a: a[:, :12])
            cands["lowlevel visual only"] = load_block("lowlevel", "timecourse", lambda a: a[:, :12, :6])
            cands["lowlevel audio only"] = load_block("lowlevel", "timecourse", lambda a: a[:, :12, 6:])
        if "ast" in avail:
            for i, l in enumerate((4, 8, 12)):
                cands[f"AST layer {l}"] = load_block("ast", "layers", lambda a, i=i: a[:, i], n_pca=128)
        if "videomae" in avail:
            for i, l in enumerate((4, 8, 12)):
                cands[f"VideoMAE layer {l}"] = load_block("videomae", "layers", lambda a, i=i: a[:, i], n_pca=128)
        if "clip_l14" in avail:
            cands["CLIP-L14 embed mean"] = load_block("clip_l14", "embed", lambda a: a.mean(1))
            cands["CLIP-L14 embed 0-1 s (6 frames)"] = load_block("clip_l14", "embed", lambda a: a[:, :6], n_pca=256)
            for i, l in enumerate((6, 12, 18, 24)):
                cands[f"CLIP-L14 layer {l} mean"] = load_block("clip_l14", "cls", lambda a, i=i: a[:, :, i].mean(1), n_pca=128)
        for name, b in cands.items():
            evaluate(concat(b), prep_kw, **ens, tag=name)
    print(f"done in {time.time() - t0:.0f} s")

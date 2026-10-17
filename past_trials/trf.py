"""Time-resolved encoding model (temporal response function, TRF).

Instead of "one summary of the whole video -> the whole EEG window" (model.py), the TRF learns
"EEG at moment t = response to what happened in the video during the last LAG_MAX seconds".
The same weights are used at every moment, so every time point of every video is a training
example, and the whole 3 s of EEG can be used.

Time grid: 20 Hz (0.05 s steps), 60 points over 0-3 s, for both EEG and video features.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from mindreading.config import CACHE_DIR, SEED, SUBJECTS
from mindreading.data import load_test, load_train
from mindreading.preprocess import prepare

HZ = 20
N_T = 3 * HZ                      # 60 time points over 3 s
ALPHAS = np.logspace(-1, 7, 17)


# ---------------------------------------------------------------- time-resolved features

def _interp(arr, src_t, n_t=N_T):
    """arr (videos, len(src_t), d) sampled at src_t -> (videos, n_t, d) on the 20 Hz grid."""
    dst_t = (np.arange(n_t) + 0.5) / HZ
    out = np.empty((arr.shape[0], n_t, arr.shape[2]), np.float32)
    for k in range(arr.shape[2]):
        out[:, :, k] = np.array([np.interp(dst_t, src_t, row) for row in arr[:, :, k]])
    return out


def time_features(blocks=("low", "clip"), clip_pca=16):
    """{video id: (60 time points, d)} standardised on training videos."""
    parts, ids = [], None
    if "low" in blocks:
        z = np.load(CACHE_DIR / "lowlevel.npz")
        ids = z["ids"].astype(int)
        tc = z["timecourse"].astype(np.float32)                    # (videos, 30 bins of 0.1 s, 8)
        parts.append(_interp(tc, (np.arange(30) + 0.5) / 10))
    if "clip" in blocks:
        z = np.load(CACHE_DIR / "clip_b32.npz")
        ids = z["ids"].astype(int) if ids is None else ids
        fr = z["frames"].astype(np.float32)                        # (videos, 8 frames, 512)
        tr = ids <= 1000
        pca = PCA(clip_pca, random_state=SEED).fit(fr[tr].reshape(-1, fr.shape[2]))
        red = pca.transform(fr.reshape(-1, fr.shape[2])).reshape(len(fr), 8, clip_pca)
        parts.append(_interp(red.astype(np.float32), np.linspace(0, 89, 8) / 29.97))
        # frame-to-frame change of the CLIP content: "something new appeared"
        d = np.linalg.norm(np.diff(fr, axis=1), axis=2)[:, :, None]
        parts.append(_interp(d.astype(np.float32), (np.linspace(0, 89, 8)[1:] + np.linspace(0, 89, 8)[:-1]) / 2 / 29.97))
    win_centres = np.arange(0.0, 2.01, 0.25) + 0.5          # sliding 1-s windows (features.py)
    for blk, file, key, src_t in (("clip_dense", "clip_dense", "frames", np.arange(15) * 0.2 + 0.1),
                                  ("mae_w", "videomae_windows", "windows", win_centres),
                                  ("ast_w", "ast_windows", "windows", win_centres)):
        if blk in blocks:
            z = np.load(CACHE_DIR / f"{file}.npz")
            assert (z["ids"].astype(int) == ids).all()
            arr = z[key].astype(np.float32)
            tr = ids <= 1000
            pca = PCA(clip_pca, random_state=SEED).fit(arr[tr].reshape(-1, arr.shape[2]))
            red = pca.transform(arr.reshape(-1, arr.shape[2])).reshape(len(arr), arr.shape[1], clip_pca)
            parts.append(_interp(red.astype(np.float32), src_t))
    X = np.concatenate(parts, axis=2)
    tr = ids <= 1000
    mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6
    X = (X - mu) / sd
    return dict(zip(ids.tolist(), X))


def lagged(F, lags):
    """F (n_t, d) -> (n_t, d * len(lags)); feature at t - lag, zero before the video starts."""
    out = np.zeros((F.shape[0], F.shape[1] * len(lags)), np.float32)
    for j, L in enumerate(lags):
        out[L:, j * F.shape[1]:(j + 1) * F.shape[1]] = F[:F.shape[0] - L]
    return out


# ---------------------------------------------------------------- EEG on the 20 Hz grid

def eeg_20hz(raw, whiten=0.3):
    """raw (n, 17, 1500) -> (n, 60, 17): 100 Hz preprocessing, then mean over blocks of 5."""
    x = prepare(raw, "set", 0.0, 3.0, False, whiten)              # (n, 17, 300)
    x = x.reshape(x.shape[0], x.shape[1], N_T, -1).mean(-1)       # (n, 17, 60)
    return x.transpose(0, 2, 1).astype(np.float32)


@dataclass
class TRFConfig:
    blocks: tuple = ("low", "clip")
    clip_pca: int = 16
    lag_max: float = 0.6               # seconds of video history used for each EEG moment
    whiten: float = 0.3
    fit_tmin: float = 0.0            # train only on EEG moments after this time (s)
    score_tmin: float = 1.0          # compare predicted vs real EEG only after this time (s)
    members: tuple = ("subject", "pooled")
    member_weights: dict = field(default_factory=lambda: {"subject": 1.0, "pooled": 0.5})


class TRF:
    def __init__(self, cfg: TRFConfig):
        self.cfg = cfg
        self.feats = time_features(cfg.blocks, cfg.clip_pca)
        self.lags = list(range(0, int(round(cfg.lag_max * HZ)) + 1))
        self.design = {v: lagged(F, self.lags) for v, F in self.feats.items()}
        self.eeg = {}

    def train_eeg(self, sub):
        if sub not in self.eeg:
            ids, raw = load_train(sub)
            self.eeg[sub] = (ids, eeg_20hz(raw, self.cfg.whiten))
        return self.eeg[sub]

    def test_eeg(self, sub):
        raw, labels = load_test(sub)
        return eeg_20hz(raw, self.cfg.whiten), labels

    def _fit(self, ids, Y):
        """ids (n,), Y (n, 60, 17): one ridge over all (video, time point) samples."""
        a = int(round(self.cfg.fit_tmin * HZ))
        X = np.concatenate([self.design[int(v)][a:] for v in ids])
        y = Y[:, a:].reshape(-1, Y.shape[2])
        mu = X.mean(0)
        r = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(X - mu, y)
        return mu, r

    def predict(self, model, ids):
        mu, r = model
        X = np.concatenate([self.design[int(v)] for v in ids])
        return r.predict(X - mu).reshape(len(ids), N_T, -1)

    def fit_shared(self, fit_ids=None):
        self.shared = {}
        if "pooled" in self.cfg.members:
            I, Y = [], []
            for s in SUBJECTS:
                ids, y = self.train_eeg(s)
                m = np.ones(len(ids), bool) if fit_ids is None else np.isin(ids, fit_ids)
                I.append(ids[m]); Y.append(y[m])
            self.shared["pooled"] = self._fit(np.concatenate(I), np.concatenate(Y))
        return self

    def fit_subject(self, sub, fit_ids=None):
        if "subject" not in self.cfg.members:
            return {}
        ids, y = self.train_eeg(sub)
        m = np.ones(len(ids), bool) if fit_ids is None else np.isin(ids, fit_ids)
        return {"subject": self._fit(ids[m], y[m])}

    def scores(self, members, eeg, candidate_ids):
        """Correlation between real EEG (n, 60, 17) and each candidate's predicted EEG."""
        b = int(round(self.cfg.score_tmin * HZ))
        real = eeg[:, b:].reshape(len(eeg), -1)
        out = {}
        for k, m in members.items():
            pred = self.predict(m, candidate_ids)[:, b:].reshape(len(candidate_ids), -1)
            out[k] = _corr(real, pred)
        out["ensemble"] = sum(self.cfg.member_weights.get(k, 1.0) * _rowz(s) for k, s in out.items())
        return out


def _rowz(a):
    a = a - a.mean(1, keepdims=True)
    return a / (a.std(1, keepdims=True) + 1e-8)


def _corr(a, b):
    return _rowz(a) @ _rowz(b).T / a.shape[1]

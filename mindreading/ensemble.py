"""Improved model: multi-feature encoder ensemble.

What changed compared with the base model (each change was validated on held-out
*training* videos, see scripts/exp_eeg.py and scripts/exp_combo.py):

1. EEG: common average reference (subtract the mean over electrodes) before z-scoring.
2. Video features: several blocks (CLIP, audio, low-level time courses, ...). Each block is
   standardised on the training videos, optionally PCA-reduced, scaled to equal total size,
   and multiplied by a block weight. Blocks are concatenated.
3. Three encoders per target subject, whose scores are z-scored and added:
   - "subject":  trained on that subject's own training recordings
   - "pooled":   trained on the training recordings of all 6 subjects
   - "ga":       trained on the grand average over subjects per video (much less noise)
Candidates are ranked by the correlation between the real EEG and each encoder's
predicted EEG for that candidate video.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .config import CACHE_DIR, SEED, SUBJECTS, TEST_IDS, TRAIN_IDS
from .data import load_test, load_train
from .models import correlation_matrix, topk
from .preprocess import prepare

ALPHAS = np.logspace(-3, 7, 21)

# name -> (cache file, array key, selection applied to the array, PCA dims or None)
BLOCKS = {
    "clip":    ("clip_b32", "mean", None, None),
    "ast4":    ("ast", "layers", lambda a: a[:, 0], 128),            # audio, AST layer 4
    "ast":     ("ast", "layers", None, 128),                          # audio, AST layers 4/8/12
    "low":     ("lowlevel", "timecourse", lambda a: a[:, :12], None),  # 0-1.2 s time courses
    "mae":     ("videomae", "layers", None, 128),                     # VideoMAE layers 4/8/12
    "l14":     ("clip_l14", "embed", lambda a: a.mean(1), None),     # CLIP-L/14, mean of 16 frames
    "l14cls":  ("clip_l14", "cls", lambda a: a.mean(1), 256),        # CLIP-L/14 layers 6/12/18/24
}


def load_block(name):
    file, key, select, n_pca = BLOCKS[name]
    z = np.load(CACHE_DIR / f"{file}.npz")
    arr = z[key].astype(np.float32)
    if select is not None:
        arr = select(arr)
    X = arr.reshape(len(arr), -1)
    ids = z["ids"].astype(int)
    tr = ids <= 1000                                  # fit scaling/PCA on training videos only
    X = StandardScaler().fit(X[tr]).transform(X)
    if n_pca and n_pca < X.shape[1]:
        X = PCA(n_pca, random_state=SEED).fit(X[tr]).transform(X)
        X /= X[tr].std(0, keepdims=True).mean()
    X /= np.sqrt(X.shape[1])
    return dict(zip(ids.tolist(), X.astype(np.float32)))


class VideoFeatures:
    def __init__(self, blocks: dict):
        """blocks: {block name: weight}."""
        self.blocks = blocks
        self.tables = {b: load_block(b) for b in blocks}

    def __call__(self, ids):
        return np.concatenate([w * np.stack([self.tables[b][int(i)] for i in ids])
                               for b, w in self.blocks.items()], 1)


@dataclass
class EnsembleConfig:
    blocks: dict = field(default_factory=lambda: {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 0.6})
    tmin: float = 0.0
    tmax: float = 1.0
    car: bool = False
    whiten: float = 0.3                # channel whitening shrinkage (0 = off)
    members: tuple = ("subject", "pooled", "ga")
    member_weights: dict = field(default_factory=lambda: {"subject": 1.0, "pooled": 0.25, "ga": 0.25})
    shuffle_labels: bool = False


class Encoder:
    """Video features -> flattened EEG, ridge regression (features are only centred).
    Each EEG feature (electrode x time point) gets its own regularisation strength, chosen by
    efficient leave-one-out CV: uninformative points get shrunk to ~0 and stop affecting the
    correlation, so the comparison focuses on the predictable parts of the response."""

    def fit(self, V, Y):
        self.mu = V.mean(0)
        self.ridge = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(V - self.mu, Y)
        return self

    def predict(self, V):
        return self.ridge.predict(V - self.mu)


def _rowz(a):
    a = a - a.mean(1, keepdims=True)
    return a / (a.std(1, keepdims=True) + 1e-8)


class EnsembleModel:
    def __init__(self, cfg: EnsembleConfig, feats: VideoFeatures | None = None):
        self.cfg = cfg
        self.feats = feats or VideoFeatures(cfg.blocks)
        self.eeg = {}

    # -------- data
    def train_eeg(self, sub):
        if sub not in self.eeg:
            ids, e = load_train(sub)
            x = prepare(e, "set", self.cfg.tmin, self.cfg.tmax, self.cfg.car, self.cfg.whiten)
            self.eeg[sub] = (ids, x.reshape(len(x), -1))
        return self.eeg[sub]

    def test_eeg(self, sub):
        e, labels = load_test(sub)
        x = prepare(e, "set", self.cfg.tmin, self.cfg.tmax, self.cfg.car, self.cfg.whiten)
        return x.reshape(len(x), -1), labels

    # -------- training
    def _encoder(self, ids, Y):
        if self.cfg.shuffle_labels:
            ids = np.random.default_rng(SEED).permutation(ids)
        return Encoder().fit(self.feats(ids), Y)

    def fit_shared(self, fit_ids=None):
        """Members shared by all subjects: pooled and grand-average encoders."""
        keep = (lambda i: np.ones(len(i), bool)) if fit_ids is None else (lambda i: np.isin(i, fit_ids))
        data = {s: self.train_eeg(s) for s in SUBJECTS}
        self.shared = {}
        if "pooled" in self.cfg.members:
            I = np.concatenate([i[keep(i)] for i, _ in data.values()])
            Y = np.concatenate([y[keep(i)] for i, y in data.values()])
            self.shared["pooled"] = self._encoder(I, Y)
        if "ga" in self.cfg.members:
            sums, counts = {}, {}
            for i, y in data.values():
                for v, row in zip(i[keep(i)], y[keep(i)]):
                    sums[v] = sums.get(v, 0) + row
                    counts[v] = counts.get(v, 0) + 1
            ga_ids = np.array(sorted(sums))
            self.shared["ga"] = self._encoder(ga_ids, np.stack([sums[v] / counts[v] for v in ga_ids]))
        return self

    def fit_subject(self, sub, fit_ids=None):
        ids, Y = self.train_eeg(sub)
        m = np.ones(len(ids), bool) if fit_ids is None else np.isin(ids, fit_ids)
        return {"subject": self._encoder(ids[m], Y[m])} if "subject" in self.cfg.members else {}

    # -------- scoring
    def scores(self, members: dict, eeg, candidate_ids):
        C = self.feats(candidate_ids)
        per = {k: correlation_matrix(eeg, m.predict(C)) for k, m in members.items()}
        per["ensemble"] = sum(self.cfg.member_weights.get(k, 1.0) * _rowz(s) for k, s in per.items())
        return per


def split_train_videos(frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = np.array(TRAIN_IDS)
    val = np.sort(rng.choice(ids, int(len(ids) * frac), replace=False))
    return np.setdiff1d(ids, val), val


def validate(cfg, seeds=(0, 1, 2), subjects=SUBJECTS, model=None):
    """Held-out TRAINING videos (no test data). Mean over seeds; returns {member: {top1, top5}}."""
    model = model or EnsembleModel(cfg)
    acc = {}
    for seed in seeds:
        fit_ids, val_ids = split_train_videos(seed=seed)
        model.fit_shared(fit_ids)
        for s in subjects:
            members = {**model.fit_subject(s, fit_ids), **model.shared}
            ids, Y = model.train_eeg(s)
            m = np.isin(ids, val_ids)
            cand = np.unique(ids[m])
            for k, sc in model.scores(members, Y[m], cand).items():
                acc.setdefault(k, []).append(topk(sc, np.searchsorted(cand, ids[m])))
    return {k: {q: float(np.mean([r[q] for r in v])) for q in ("top1", "top5", "median_rank")}
            for k, v in acc.items()}


def dev_test(cfg, subjects=(1, 2, 3, 4), model=None):
    """Train on all training recordings; score the sub-01..04 test sets (report only)."""
    model = model or EnsembleModel(cfg)
    model.fit_shared()
    cand = np.array(TEST_IDS)
    res = {}
    for s in subjects:
        members = {**model.fit_subject(s), **model.shared}
        eeg, labels = model.test_eeg(s)
        res[s] = {k: topk(sc, labels - cand[0]) for k, sc in model.scores(members, eeg, cand).items()}
    return res


def predict_ranked(model, sub, members):
    """Score matrix (102 recordings in original order x 102 candidates) for a subject."""
    eeg, _ = model.test_eeg(sub)
    return model.scores(members, eeg, np.array(TEST_IDS))["ensemble"]

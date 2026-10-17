"""The model: predict the EEG each video should cause, then rank videos by similarity.

How it works (every choice validated on held-out *training* videos, see past_trials/experiment_log.md):

1. Video features ("cards"): four blocks per video, each standardised on the training videos,
   optionally PCA-reduced, scaled to equal size, weighted, and concatenated:
       clip  (1.0)  CLIP ViT-B/32, what is in the picture
       mae   (0.6)  VideoMAE, motion and actions
       low   (0.6)  brightness / motion / cuts / loudness ... every 0.1 s
       ast4  (0.5)  AST audio model, what the sound is
2. EEG: 100 Hz, channel whitening, z-score per file, 0-1 s after video onset (preprocess.py).
3. Encoder: ridge regression from the video features to the EEG (17 electrodes x 100 time
   points). Every electrode x time point gets its own regularisation strength, so parts of
   the EEG that can't be predicted are shrunk to ~0.
4. Three encoders per subject, scores added with weights:
       subject (1.0)  trained on that subject's own training recordings
       pooled  (0.25) trained on all 6 subjects' training recordings
       ga      (0.25) trained on the grand-average EEG over subjects for each video
5. Score of (recording, candidate video) = correlation between the real EEG and the EEG the
   encoder predicts for that video. Higher = more likely.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .config import CACHE_DIR, SEED, SUBJECTS, TEST_IDS, TRAIN_IDS
from .data import load_test, load_train
from .preprocess import band_power, prepare

ALPHAS = np.logspace(-3, 7, 21)   # ridge strengths to choose from

# block name -> (cache file, array key, selection applied to the array, PCA dims or None)
BLOCKS = {
    "clip": ("clip_b32", "mean", None, None),                        # CLIP, 8 frames averaged
    "mae":  ("videomae", "layers", None, 128),                       # VideoMAE layers 4/8/12
    "low":  ("lowlevel", "timecourse", lambda a: a[:, :12], None),   # time courses, 0-1.2 s
    "ast4": ("ast", "layers", lambda a: a[:, 0], 128),               # AST audio, layer 4
}


# ---------------------------------------------------------------- video features

def load_block(name):
    """{video id: feature vector} for one block, scaled using training videos only."""
    file, key, select, n_pca = BLOCKS[name]
    z = np.load(CACHE_DIR / f"{file}.npz")
    arr = z[key].astype(np.float32)
    if select is not None:
        arr = select(arr)
    X = arr.reshape(len(arr), -1)
    ids = z["ids"].astype(int)
    tr = ids <= 1000
    X = StandardScaler().fit(X[tr]).transform(X)
    if n_pca and n_pca < X.shape[1]:
        X = PCA(n_pca, random_state=SEED).fit(X[tr]).transform(X)
        X /= X[tr].std(0, keepdims=True).mean()
    X /= np.sqrt(X.shape[1])
    return dict(zip(ids.tolist(), X.astype(np.float32)))


class VideoFeatures:
    """Call with a list of video ids -> matrix (videos x all weighted blocks)."""

    def __init__(self, blocks: dict):
        self.blocks = blocks
        self.tables = {b: load_block(b) for b in blocks}

    def __call__(self, ids):
        return np.concatenate([w * np.stack([self.tables[b][int(i)] for i in ids])
                               for b, w in self.blocks.items()], 1)


# ---------------------------------------------------------------- model

@dataclass
class Config:
    blocks: dict = field(default_factory=lambda: {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 0.6})
    tmin: float = 0.0                  # EEG window, seconds after video onset
    tmax: float = 1.0
    car: bool = False                  # common average reference (replaced by whitening)
    whiten: float = 0.3                # channel whitening shrinkage (0 = off)
    members: tuple = ("subject", "pooled", "ga")
    member_weights: dict = field(default_factory=lambda: {"subject": 1.0, "pooled": 0.25, "ga": 0.25})
    power_weight: float = 0.25         # weight of the rhythm-power member (0 = off)
    shuffle_labels: bool = False       # sanity check: wrong video per recording -> must score chance


class Encoder:
    """Video features -> flattened EEG (ridge regression, one strength per EEG feature)."""

    def fit(self, V, Y):
        self.mu = V.mean(0)
        self.ridge = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(V - self.mu, Y)
        return self

    def predict(self, V):
        return self.ridge.predict(V - self.mu)


def _rowz(a):
    a = a - a.mean(1, keepdims=True)
    return a / (a.std(1, keepdims=True) + 1e-8)


def correlation_matrix(a, b):
    """Pearson correlation between every row of a and every row of b."""
    return _rowz(a) @ _rowz(b).T / a.shape[1]


class Model:
    def __init__(self, cfg: Config, feats: VideoFeatures | None = None):
        self.cfg = cfg
        self.feats = feats or VideoFeatures(cfg.blocks)
        self.eeg = {}

    # -------- data
    def train_eeg(self, sub):
        """(video ids, EEG matrix recordings x 1700) for a subject's training file, cached."""
        if sub not in self.eeg:
            ids, e = load_train(sub)
            x = prepare(e, "set", self.cfg.tmin, self.cfg.tmax, self.cfg.car, self.cfg.whiten)
            self.eeg[sub] = (ids, x.reshape(len(x), -1))
        return self.eeg[sub]

    def train_power(self, sub):
        """Rhythm power (recordings x 17*5*6) for a subject's training file, same row order."""
        key = ("power", sub)
        if key not in self.eeg:
            _, e = load_train(sub)
            self.eeg[key] = band_power(e, self.cfg.whiten)
        return self.eeg[key]

    def test_eeg(self, sub):
        """(EEG matrix 102 x 1700 in the file's original order, true ids or None)."""
        e, labels = load_test(sub)
        x = prepare(e, "set", self.cfg.tmin, self.cfg.tmax, self.cfg.car, self.cfg.whiten)
        return x.reshape(len(x), -1), labels

    def test_power(self, sub):
        e, _ = load_test(sub)
        return band_power(e, self.cfg.whiten)

    # -------- training
    def _encoder(self, ids, Y):
        if self.cfg.shuffle_labels:
            ids = np.random.default_rng(SEED).permutation(ids)
        return Encoder().fit(self.feats(ids), Y)

    def fit_shared(self, fit_ids=None):
        """Encoders shared by all subjects: pooled and grand average."""
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
        """The subject's own encoders: EEG wave (0-1 s) and, if enabled, rhythm power (0-3 s)."""
        ids, Y = self.train_eeg(sub)
        m = np.ones(len(ids), bool) if fit_ids is None else np.isin(ids, fit_ids)
        out = {"subject": self._encoder(ids[m], Y[m])} if "subject" in self.cfg.members else {}
        if self.cfg.power_weight:
            out["power"] = self._encoder(ids[m], self.train_power(sub)[m])
        return out

    # -------- scoring
    def scores(self, members: dict, eeg, candidate_ids, power=None):
        """{member: score matrix, 'wave': weighted wave members, 'ensemble': final}, each
        (recordings x candidates). `power` = rhythm power of the same recordings."""
        C = self.feats(candidate_ids)
        per = {k: correlation_matrix(eeg, m.predict(C)) for k, m in members.items() if k != "power"}
        per["wave"] = sum(self.cfg.member_weights.get(k, 1.0) * _rowz(s) for k, s in per.items())
        per["ensemble"] = per["wave"]
        if "power" in members and power is not None:
            per["power"] = correlation_matrix(power, members["power"].predict(C))
            per["ensemble"] = _rowz(per["wave"]) + self.cfg.power_weight * _rowz(per["power"])
        return per


# ---------------------------------------------------------------- evaluation

def topk(scores, true_cols):
    """top-1, top-5 and median rank of the true candidate (rank 1 = best)."""
    true_score = scores[np.arange(len(scores)), true_cols]
    r = (scores > true_score[:, None]).sum(axis=1) + 1
    return {"top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()), "median_rank": float(np.median(r))}


def split_train_videos(frac=0.15, seed=SEED):
    """Hide a random 15% of the training videos (same videos for every subject)."""
    rng = np.random.default_rng(seed)
    ids = np.array(TRAIN_IDS)
    val = np.sort(rng.choice(ids, int(len(ids) * frac), replace=False))
    return np.setdiff1d(ids, val), val


def validate(cfg, seeds=(0, 1, 2), subjects=SUBJECTS, model=None):
    """Score on hidden TRAINING videos (no test data), averaged over 3 random splits."""
    model = model or Model(cfg)
    acc = {}
    for seed in seeds:
        fit_ids, val_ids = split_train_videos(seed=seed)
        model.fit_shared(fit_ids)
        for s in subjects:
            members = {**model.fit_subject(s, fit_ids), **model.shared}
            ids, Y = model.train_eeg(s)
            m = np.isin(ids, val_ids)
            cand = np.unique(ids[m])
            P = model.train_power(s)[m] if cfg.power_weight else None
            for k, sc in model.scores(members, Y[m], cand, P).items():
                acc.setdefault(k, []).append(topk(sc, np.searchsorted(cand, ids[m])))
    return {k: {q: float(np.mean([r[q] for r in v])) for q in ("top1", "top5", "median_rank")}
            for k, v in acc.items()}


def dev_test(cfg, subjects=(1, 2, 3, 4), model=None):
    """Train on all training recordings; score the sub-01..04 test sets (report only)."""
    model = model or Model(cfg)
    model.fit_shared()
    cand = np.array(TEST_IDS)
    res = {}
    for s in subjects:
        members = {**model.fit_subject(s), **model.shared}
        eeg, labels = model.test_eeg(s)
        P = model.test_power(s) if cfg.power_weight else None
        res[s] = {k: topk(sc, labels - cand[0]) for k, sc in model.scores(members, eeg, cand, P).items()}
    return res


def summarise(res):
    """Mean over subjects for each member."""
    members = next(iter(res.values())).keys()
    return {m: {k: float(np.mean([r[m][k] for r in res.values()])) for k in ("top1", "top5", "median_rank")}
            for m in members}


def predict_scores(model, sub, members):
    """Score matrix for a subject's test file: 102 recordings (original order) x 102 candidates."""
    eeg, _ = model.test_eeg(sub)
    P = model.test_power(sub) if model.cfg.power_weight else None
    return model.scores(members, eeg, np.array(TEST_IDS), P)["ensemble"]

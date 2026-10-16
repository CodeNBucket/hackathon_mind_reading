"""End-to-end: load -> preprocess -> train -> score candidates.

Two ways to evaluate a configuration:
- `validate_on_train_split`: hold out 15% of the *training videos* (same videos for every
  subject), train on the rest, rank the held-out videos. Used to choose settings, as the
  rules ask.
- `score_dev_test`: train on all training recordings, rank the 102 test videos for the
  sub-01..04 test recordings, compare with tst_stim. Used to report the score only.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import SEED, SUBJECTS, TEST_IDS, TRAIN_IDS
from .data import load_test, load_train
from .models import Decoder, Encoder, combine, topk
from .preprocess import prepare
from .video_features import lookup


@dataclass
class Config:
    tmin: float = 0.0
    tmax: float = 1.0
    norm: str = "set"
    feature_key: str = "mean"         # "mean" (512) or "frames" (8 x 512 flattened)
    pool_subjects: bool = True         # train on all 6 subjects' training recordings
    methods: tuple = ("encoder", "decoder")
    enc_pca: int | None = None
    dec_pca: int | None = 200
    train_subjects: list = field(default_factory=lambda: list(SUBJECTS))
    shuffle_labels: bool = False       # sanity check: wrong video per recording -> must score chance


def video_matrix(features, ids, key):
    x = lookup(features, ids, key)
    return x.reshape(len(x), -1)


class Pipeline:
    def __init__(self, cfg: Config, features: dict):
        self.cfg, self.features = cfg, features
        self.cache = {}

    # ---- data -------------------------------------------------------------
    def train_data(self, sub):
        if ("trn", sub) not in self.cache:
            ids, eeg = load_train(sub)
            self.cache[("trn", sub)] = (ids, prepare(eeg, self.cfg.norm, self.cfg.tmin, self.cfg.tmax))
        return self.cache[("trn", sub)]

    def test_data(self, sub):
        if ("tst", sub) not in self.cache:
            eeg, labels = load_test(sub)
            self.cache[("tst", sub)] = (prepare(eeg, self.cfg.norm, self.cfg.tmin, self.cfg.tmax), labels)
        return self.cache[("tst", sub)]

    def stacked(self, subs, keep_ids=None):
        X, ids = [], []
        for s in subs:
            i, e = self.train_data(s)
            m = np.ones(len(i), bool) if keep_ids is None else np.isin(i, keep_ids)
            X.append(e[m]); ids.append(i[m])
        return np.concatenate(X), np.concatenate(ids)

    # ---- training and scoring -------------------------------------------
    def fit(self, sub, keep_ids=None):
        subs = self.cfg.train_subjects if self.cfg.pool_subjects else [sub]
        eeg, ids = self.stacked(subs, keep_ids)
        if self.cfg.shuffle_labels:
            ids = np.random.default_rng(SEED).permutation(ids)
        V = video_matrix(self.features, ids, self.cfg.feature_key)
        models = {}
        if "encoder" in self.cfg.methods:
            models["encoder"] = Encoder(self.cfg.enc_pca).fit(V, eeg)
        if "decoder" in self.cfg.methods:
            models["decoder"] = Decoder(self.cfg.dec_pca).fit(eeg, V)
        return models

    def scores(self, models, eeg, candidate_ids):
        C = video_matrix(self.features, candidate_ids, self.cfg.feature_key)
        out = {name: m.scores(eeg, C) for name, m in models.items()}
        if len(out) > 1:
            out["combined"] = combine(*out.values())
        return out


def split_train_videos(frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = np.array(TRAIN_IDS)
    val = np.sort(rng.choice(ids, int(len(ids) * frac), replace=False))
    return np.setdiff1d(ids, val), val


def validate_on_train_split(cfg, features, subjects=SUBJECTS, pipe=None):
    """Score per subject (and mean) when ranking held-out training videos. Uses no test data."""
    pipe = pipe or Pipeline(cfg, features)
    fit_ids, val_ids = split_train_videos()
    res = {}
    shared = pipe.fit(None, fit_ids) if cfg.pool_subjects else None
    for s in subjects:
        models = shared or pipe.fit(s, fit_ids)
        ids, eeg = pipe.train_data(s)
        m = np.isin(ids, val_ids)
        cand = np.unique(ids[m])
        sc = pipe.scores(models, eeg[m], cand)
        true_cols = np.searchsorted(cand, ids[m])
        res[s] = {k: topk(v, true_cols) for k, v in sc.items()}
    return res


def score_dev_test(cfg, features, subjects=(1, 2, 3, 4), pipe=None, return_scores=False):
    """Train on all training recordings, rank the 102 test videos for sub-01..04."""
    pipe = pipe or Pipeline(cfg, features)
    cand = np.array(TEST_IDS)
    shared = pipe.fit(None) if cfg.pool_subjects else None
    res, raw = {}, {}
    for s in subjects:
        models = shared or pipe.fit(s)
        eeg, labels = pipe.test_data(s)
        sc = pipe.scores(models, eeg, cand)
        true_cols = labels - cand[0]
        res[s] = {k: topk(v, true_cols) for k, v in sc.items()}
        raw[s] = sc
    return (res, raw) if return_scores else res


def summarise(res):
    """Mean over subjects for each method."""
    methods = next(iter(res.values())).keys()
    return {m: {k: float(np.mean([r[m][k] for r in res.values()])) for k in ("top1", "top5", "median_rank")}
            for m in methods}

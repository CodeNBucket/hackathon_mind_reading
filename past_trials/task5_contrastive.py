"""Task 4b: refine the winning encoder (ridge + contrastive MLP).

  - combine with the Task 3 weights (VideoMAE 1.5)
  - contrastive temperature 0.05 vs 0.1, hidden size
  - average 3 networks trained with different seeds (neural nets vary run to run)
Then, for the best variant: the sub-01..04 test score (report only) and the shuffled-label control.

    .venv\\Scripts\\python.exe past_trials\\task5_contrastive.py
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mindreading.model as M  # noqa: E402
import task4_encoders as T4  # noqa: E402
from mindreading.config import SEED, SUBJECTS, TEST_IDS  # noqa: E402
from mindreading.model import Config, Model, _rowz, topk  # noqa: E402

W_MAE15 = {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 1.5}


class SeedAvg:
    """Average the predictions of n contrastive nets trained with different seeds."""

    def __init__(self, n=3, **kw):
        self.n, self.kw = n, kw

    def fit(self, V, Y, ids):
        self.nets = []
        for k in range(self.n):
            T4.SEED = SEED + k                        # TorchEnc reads the module-level seed
            self.nets.append(T4.TorchEnc("contrastive", **self.kw).fit(V, Y, ids))
        T4.SEED = SEED
        return self

    def predict(self, V):
        return np.mean([n.predict(V) for n in self.nets], axis=0)


def make(factory, cfg):
    model = Model(cfg)

    def enc(self, ids, Y, _f=factory):
        if self.cfg.shuffle_labels:
            ids = np.random.default_rng(SEED).permutation(ids)
        V = self.feats(ids)
        if Y.shape[1] == T4.WAVE_DIM and _f is not None:
            return _f().fit(V, Y, np.asarray(ids))
        return M.Encoder().fit(V, Y)

    model._encoder = enc.__get__(model)
    return model


def combine(a, b, w=1.0):
    return {k: (_rowz(a[k][0]) + w * _rowz(b[k][0]), a[k][1]) for k in a}


def test_scores(model):
    """{sub: (ensemble scores, true cols)} on the sub-01..04 test sets (report only)."""
    model.fit_shared()
    cand = np.array(TEST_IDS)
    out = {}
    for s in (1, 2, 3, 4):
        members = {**model.fit_subject(s), **model.shared}
        eeg, labels = model.test_eeg(s)
        sc = model.scores(members, eeg, cand, model.test_power(s))
        out[s] = (sc["ensemble"], labels - cand[0])
    return out


def line(tag, v):
    print(f"{tag:52s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  median {v['median_rank']:.1f}", flush=True)


if __name__ == "__main__":
    res = {}
    cfg0, cfg15 = Config(), Config(blocks=W_MAE15)
    runs = {
        "ridge, weights default": (None, cfg0),
        "ridge, VideoMAE 1.5": (None, cfg15),
        "contrastive tau 0.05, weights default": (lambda: T4.TorchEnc("contrastive"), cfg0),
        "contrastive tau 0.05, VideoMAE 1.5": (lambda: T4.TorchEnc("contrastive"), cfg15),
        "contrastive tau 0.1, VideoMAE 1.5": (lambda: T4.TorchEnc("contrastive", tau=0.1), cfg15),
        "contrastive hidden 2048, VideoMAE 1.5": (lambda: T4.TorchEnc("contrastive", hidden=2048), cfg15),
        "contrastive 3-seed average, VideoMAE 1.5": (lambda: SeedAvg(3), cfg15),
    }
    cache = {}
    for tag, (f, cfg) in runs.items():
        t0 = time.time()
        cache[tag] = T4.val_scores(make(f, cfg))
        res[tag] = T4.summary(cache[tag])
        line(f"{tag} ({time.time() - t0:.0f} s)", res[tag])
    combos = [("ridge + contrastive, weights default", "ridge, weights default", "contrastive tau 0.05, weights default"),
              ("ridge + contrastive, VideoMAE 1.5", "ridge, VideoMAE 1.5", "contrastive tau 0.05, VideoMAE 1.5"),
              ("ridge + contrastive tau 0.1, VideoMAE 1.5", "ridge, VideoMAE 1.5", "contrastive tau 0.1, VideoMAE 1.5"),
              ("ridge + contrastive h2048, VideoMAE 1.5", "ridge, VideoMAE 1.5", "contrastive hidden 2048, VideoMAE 1.5"),
              ("ridge + contrastive 3 seeds, VideoMAE 1.5", "ridge, VideoMAE 1.5", "contrastive 3-seed average, VideoMAE 1.5")]
    for tag, a, b in combos:
        for w in (0.5, 1.0, 2.0):
            res[f"{tag}, w {w}"] = T4.summary(combine(cache[a], cache[b], w))
            line(f"{tag}, w {w}", res[f"{tag}, w {w}"])

    # pick the best combination on held-out training videos (top-5, then top-1)
    best = max((k for k in res if k.startswith("ridge + ")), key=lambda k: (res[k]["top5"], res[k]["top1"]))
    print(f"\nBEST on held-out training videos: {best}", flush=True)
    tag, rest = best.split(", w ")
    w = float(rest)
    a, b = next((a, b) for t, a, b in combos if t == tag)
    fa, ca = runs[a]
    fb, cb = runs[b]
    for label, shuffle in (("TEST sub-01..04", False), ("SHUFFLED control, test", True)):
        ta = test_scores(make(fa, replace(ca, shuffle_labels=shuffle)))
        tb = test_scores(make(fb, replace(cb, shuffle_labels=shuffle)))
        r_ridge = T4.summary(ta)
        r_comb = T4.summary(combine(ta, tb, w))
        res[f"{label}: ridge part"], res[f"{label}: best combination"] = r_ridge, r_comb
        line(f"{label}: ridge part", r_ridge)
        line(f"{label}: best combination", r_comb)
        if not shuffle:
            per = {s: topk(_rowz(ta[s][0]) + w * _rowz(tb[s][0]), ta[s][1]) for s in ta}
            res[f"{label}: per subject"] = per
            print("   per subject (top1/top5):", {s: (round(v['top1'], 3), round(v['top5'], 3)) for s, v in per.items()},
                  flush=True)
    res["best"] = best
    (Path(__file__).resolve().parent / "task5_contrastive_result.json").write_text(json.dumps(res, indent=2))

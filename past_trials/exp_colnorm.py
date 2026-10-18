"""Does normalising the score matrix per candidate video ("hubness" fix) improve the ranking?

    python past_trials\\exp_colnorm.py

The problem: some candidate videos get a high predicted-EEG correlation with almost EVERY
recording (they are "hubs"), so they crowd into everyone's top-5 and push the true video down.
Each test video was watched exactly once per subject, so every candidate should be the right
answer for exactly one recording. Normalising each candidate's column removes its overall bias.

Methods, applied to the final ensemble score matrix (recordings x candidates):
  none     current ranking (each row sorted as it is)
  col      z-score each column: how much better does this candidate fit THIS recording
           than it fits the other recordings?
  rowcol   alternate column and row z-scoring 5 times (balances both directions)
  csls     cross-domain similarity local scaling (Conneau et al., 2018), k = 10:
           2*S - (mean top-10 of the row) - (mean top-10 of the column)
  col_train  the CLEARLY ALLOWED version: each candidate's "usual" score (mean and spread) is
           measured against this subject's TRAINING recordings, and the test score is
           z-scored with those numbers. Uses no test recording other than the one being
           ranked, so it works one recording at a time, as the brief's expected output asks.
           In validation the reference is the subject's fitting recordings (never the hidden ones).

Same model, same features, same protocol as mindreading.model.validate (15% of training videos
hidden, 3 splits). In each split every subject watched every hidden video once, so the hidden
block is one-to-one like the real 102 x 102 test, and the methods can be checked fairly.

RULES: col / rowcol / csls use all test recordings of a subject together (unsupervised, no
labels). That is close to "unsupervised normalization of the test recordings" but not literally
it, and it cannot rank a single recording, so ask the organisers before submitting with them.
col_train uses only training recordings as the reference: safe to submit.

Results go to results/colnorm.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mindreading import model as M  # noqa: E402
from mindreading.config import RESULTS_DIR, SUBJECTS, check_data_dir  # noqa: E402


# ---------------------------------------------------------------- the normalisations

def _z(a, axis):
    return (a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-8)


def col(S):
    return _z(S, 0)


def rowcol(S, n=5):
    for _ in range(n):
        S = _z(_z(S, 0), 1)
    return S


def csls(S, k=10):
    k = min(k, min(S.shape) - 1)
    r = np.sort(S, 1)[:, -k:].mean(1, keepdims=True)    # how well each recording fits its best candidates
    c = np.sort(S, 0)[-k:, :].mean(0, keepdims=True)    # how well each candidate fits its best recordings
    return 2 * S - r - c


def col_train(S, R):
    """z-score each candidate column with its mean/std over the reference (training) recordings R."""
    return (S - R.mean(0, keepdims=True)) / (R.std(0, keepdims=True) + 1e-8)


# every method gets S (recordings x candidates) and R (training reference recordings x candidates)
METHODS = {"none": lambda S, R: S, "col": lambda S, R: col(S), "rowcol": lambda S, R: rowcol(S),
           "csls": lambda S, R: csls(S), "col_train": col_train}


# ---------------------------------------------------------------- validation

def main():
    check_data_dir()
    cfg = M.Config()
    model = M.Model(cfg)                     # final model, final settings
    rows = []
    for seed in (0, 1, 2):
        t0 = time.time()
        fit_ids, val_ids = M.split_train_videos(seed=seed)
        model.fit_shared(fit_ids)
        per = {m: [] for m in METHODS}
        for s in SUBJECTS:
            members = {**model.fit_subject(s, fit_ids), **model.shared}
            ids, Y = model.train_eeg(s)
            msk = np.isin(ids, val_ids)
            cand = np.unique(ids[msk])
            S = model.scores(members, Y[msk], cand)["ensemble"]
            R = model.scores(members, Y[~msk], cand)["ensemble"]   # training recordings only
            true_cols = np.searchsorted(cand, ids[msk])
            for m, f in METHODS.items():
                per[m].append(M.topk(f(S, R), true_cols))
        for m in METHODS:
            r = {q: float(np.mean([x[q] for x in per[m]])) for q in ("top1", "top5", "median_rank")}
            rows.append({"method": m, "seed": seed, **r})
            print(f"{m:7s} split {seed}: top-1 {r['top1']:5.1%}  top-5 {r['top5']:5.1%}  "
                  f"median {r['median_rank']:5.1f}", flush=True)
        print(f"  (split {seed} took {time.time() - t0:.0f} s)\n", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "colnorm.json").write_text(json.dumps(rows, indent=1))

    base = {d["seed"]: d for d in rows if d["method"] == "none"}
    print("method  | top-1 | top-5 | median | top-5 gain over 'none' per split")
    for m in METHODS:
        ds = [d for d in rows if d["method"] == m]
        avg = {q: np.mean([d[q] for d in ds]) for q in ("top1", "top5", "median_rank")}
        gains = " ".join(f"{100 * (d['top5'] - base[d['seed']]['top5']):+.1f}" for d in ds)
        print(f"{m:7s} | {avg['top1']:5.1%} | {avg['top5']:5.1%} | {avg['median_rank']:6.1f} | {gains}")
    print("\nA real improvement: positive gain on EVERY split, and more than ~0.5 points on average.")


if __name__ == "__main__":
    main()

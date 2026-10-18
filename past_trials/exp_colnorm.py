"""Does normalising each candidate video's scores ("hubness" fix) improve the ranking?

    python past_trials\\exp_colnorm.py

The problem: some candidate videos get a high score with almost EVERY recording (they are
"hubs"), so they crowd into everyone's top-5 and push the true video down.

Version 2: scores the FULL current model (wave members + rhythm-power member) through
Model.ranked_scores, i.e. exactly what the pipeline uses. Version 1 (git history) scored the
model without the power member and found +2.2 points of top-5; this checks whether that holds.

Methods, applied to the final ensemble score matrix S (recordings x candidates):
  raw        current ranking, no correction
  colnorm    THE PIPELINE'S METHOD (Config.colnorm): each candidate's scores z-scored with its
             mean and spread over this subject's TRAINING recordings (in validation: the
             fitting recordings only, never the hidden ones). Ranks one recording at a time.
  col_test   for comparison only, NOT for submission: the same z-scoring, but with the mean and
             spread taken over the other hidden/test recordings. Needs all test recordings at
             once, so it does not fit the brief's "given an EEG recording" pipeline.

Protocol = mindreading.model.validate: 15% of training videos hidden, 3 splits; no test data.
Decision rule: keep colnorm only if it beats raw on EVERY split (for top-1 and/or top-5).
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

METHODS = ("raw", "colnorm", "col_test")


def col_test(S):
    return (S - S.mean(0, keepdims=True)) / (S.std(0, keepdims=True) + 1e-8)


def main():
    check_data_dir()
    cfg = M.Config()
    model = M.Model(cfg)
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
            P = model.train_power(s)[msk] if cfg.power_weight else None
            sc = model.ranked_scores(members, s, Y[msk], cand, P, fit_ids)
            sc["col_test"] = col_test(sc["raw"])
            true_cols = np.searchsorted(cand, ids[msk])
            for m in METHODS:
                per[m].append(M.topk(sc[m], true_cols))
        for m in METHODS:
            r = {q: float(np.mean([x[q] for x in per[m]])) for q in ("top1", "top5", "median_rank")}
            rows.append({"method": m, "seed": seed, **r})
            print(f"{m:8s} split {seed}: top-1 {r['top1']:5.1%}  top-5 {r['top5']:5.1%}  "
                  f"median {r['median_rank']:5.1f}", flush=True)
        print(f"  (split {seed} took {time.time() - t0:.0f} s)\n", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "colnorm.json").write_text(json.dumps(
        {"note": "full model incl. rhythm-power member; validate() protocol, 3 splits", "rows": rows}, indent=1))

    base = {d["seed"]: d for d in rows if d["method"] == "raw"}
    print("method   | top-1 | top-5 | median | top-1 gain per split | top-5 gain per split")
    for m in METHODS:
        ds = [d for d in rows if d["method"] == m]
        avg = {q: np.mean([d[q] for d in ds]) for q in ("top1", "top5", "median_rank")}
        g1 = " ".join(f"{100 * (d['top1'] - base[d['seed']]['top1']):+.1f}" for d in ds)
        g5 = " ".join(f"{100 * (d['top5'] - base[d['seed']]['top5']):+.1f}" for d in ds)
        print(f"{m:8s} | {avg['top1']:5.1%} | {avg['top5']:5.1%} | {avg['median_rank']:6.1f} | {g1:20s} | {g5}")
    print("\nKeep colnorm only if its gains are positive on EVERY split (top-1 and/or top-5).")


if __name__ == "__main__":
    main()

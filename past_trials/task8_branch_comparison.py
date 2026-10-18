"""Compare branches on 5 fresh splits (seeds 3-7) and the sub-01..04 test sets.

    python compare.py <repo path> <label> [colnorm]

With 'colnorm' (for overnight), İrem's column normalisation is applied on top: each candidate's
ensemble score is z-scored with its mean/std over the subject's TRAINING recordings (fit videos
only during validation), exactly as on the column-normalisation branch. Raw and colnorm are
reported from the same trained models.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from mindreading import model as M

label = sys.argv[2]
own_colnorm = len(sys.argv) > 3 and sys.argv[3] == "colnorm"
SEEDS = (3, 4, 5, 6, 7)
cfg = M.Config()
mdl = M.Model(cfg)
has_power = getattr(cfg, "power_weight", 0)


def score(members, eeg, cand, P):
    return mdl.scores(members, eeg, cand, P) if has_power else mdl.scores(members, eeg, cand)


def with_colnorm(members, s, eeg, cand, P, fit_ids=None):
    per = score(members, eeg, cand, P)
    ids, Y = mdl.train_eeg(s)
    m = np.ones(len(ids), bool) if fit_ids is None else np.isin(ids, fit_ids)
    R = score(members, Y[m], cand, mdl.train_power(s)[m] if has_power else None)["ensemble"]
    return {"raw": per["ensemble"], "colnorm": (per["ensemble"] - R.mean(0)) / (R.std(0) + 1e-8)}


t0 = time.time()
if own_colnorm:
    acc = {}
    for seed in SEEDS:
        fit_ids, val_ids = M.split_train_videos(seed=seed)
        mdl.fit_shared(fit_ids)
        for s in M.SUBJECTS:
            members = {**mdl.fit_subject(s, fit_ids), **mdl.shared}
            ids, Y = mdl.train_eeg(s)
            m = np.isin(ids, val_ids)
            cand = np.unique(ids[m])
            P = mdl.train_power(s)[m] if has_power else None
            for k, sc in with_colnorm(members, s, Y[m], cand, P, fit_ids).items():
                acc.setdefault(k, []).append(M.topk(sc, np.searchsorted(cand, ids[m])))
    val = {k: {q: float(np.mean([r[q] for r in v])) for q in ("top1", "top5", "median_rank")} for k, v in acc.items()}
    mdl.fit_shared()
    cand = np.array(M.TEST_IDS)
    res = {}
    for s in (1, 2, 3, 4):
        members = {**mdl.fit_subject(s), **mdl.shared}
        eeg, labels = mdl.test_eeg(s)
        P = mdl.test_power(s) if has_power else None
        res[s] = {k: M.topk(sc, labels - cand[0]) for k, sc in with_colnorm(members, s, eeg, cand, P).items()}
    test = M.summarise(res)
else:
    val = M.validate(cfg, seeds=SEEDS, model=mdl)
    test = M.summarise(M.dev_test(cfg, model=mdl))

keys = [k for k in ("ensemble", "raw", "colnorm") if k in val]
for k in keys:
    v, t = val[k], test[k]
    print(f"{label} [{k}]  fresh splits: top1 {v['top1']:.3f} top5 {v['top5']:.3f} median {v['median_rank']:.1f} | "
          f"test: top1 {t['top1']:.3f} top5 {t['top5']:.3f} median {t['median_rank']:.1f}", flush=True)
json.dump({"val": val, "test": test}, open(sys.argv[0].replace("compare.py", f"compare_{label}.json"), "w"), indent=1)
print(f"({time.time() - t0:.0f} s)")

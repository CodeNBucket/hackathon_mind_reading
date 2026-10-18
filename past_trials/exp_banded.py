"""Banded ridge on vs off, full model (Config.banded, see BandedEncoder in mindreading/model.py).

    python past_trials\\exp_banded.py            # full model: ridge + contrastive + power + colnorm
    python past_trials\\exp_banded.py --fast     # contrastive off in both (ridge part only, quicker)
    python past_trials\\exp_banded.py --seeds 3,4,5,6,7   # the team's fresh splits

Same protocol as mindreading.model.validate (15% of training videos hidden per split). The
scales are chosen inside each encoder on its own training videos, never on the hidden ones.
Prints per-split scores, the gains, and the learned scale per block. -> results/banded.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mindreading import model as M  # noqa: E402
from mindreading.config import RESULTS_DIR, SUBJECTS, check_data_dir  # noqa: E402


def run_split(model, cfg, seed, scales):
    fit_ids, val_ids = M.split_train_videos(seed=seed)
    model.fit_shared(fit_ids)
    per = []
    for s in SUBJECTS:
        members = {**model.fit_subject(s, fit_ids), **model.shared}
        scales += [(k, e.scales) for k, e in members.items() if isinstance(e, M.BandedEncoder)
                   and (k not in model.shared or s == SUBJECTS[0])]
        ids, Y = model.train_eeg(s)
        m = np.isin(ids, val_ids)
        cand = np.unique(ids[m])
        P = model.train_power(s)[m] if cfg.power_weight else None
        per.append(M.topk(model.ranked_scores(members, s, Y[m], cand, P, fit_ids)["ensemble"],
                          np.searchsorted(cand, ids[m])))
    return {q: float(np.mean([r[q] for r in per])) for q in ("top1", "top5", "median_rank")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="contrastive network off in both variants")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    check_data_dir()

    base = M.Config()
    if a.fast:
        base = replace(base, contrastive_weight=0.0)
    feats = M.VideoFeatures(base.blocks)
    eeg = {}
    rows, scales = [], []
    seeds = [int(x) for x in a.seeds.split(",")]
    for seed in seeds:
        for banded in (False, True):
            t0 = time.time()
            cfg = replace(base, banded=banded)
            model = M.Model(cfg, feats)
            model.eeg = eeg                                   # share preprocessed EEG
            r = run_split(model, cfg, seed, scales)
            rows.append({"banded": banded, "seed": seed, **r})
            print(f"banded={str(banded):5s} split {seed}: top-1 {r['top1']:5.1%}  top-5 {r['top5']:5.1%}  "
                  f"median {r['median_rank']:5.1f}  ({time.time() - t0:.0f} s)", flush=True)

    blocks = list(base.blocks)
    mean_scale = {b: float(np.exp(np.mean([np.log(s[b]) for _, s in scales]))) for b in blocks} if scales else {}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "banded.json").write_text(json.dumps(
        {"fast": a.fast, "rows": rows, "mean_scale": mean_scale, "start_weights": base.blocks,
         "learned_scales": [{"encoder": k, **s} for k, s in scales]}, indent=1))

    off = {d["seed"]: d for d in rows if not d["banded"]}
    print(f"\n{'model':9s} | top-1 | top-5 | median | top-1 gain per split | top-5 gain per split")
    for banded in (False, True):
        ds = [d for d in rows if d["banded"] == banded]
        avg = {q: np.mean([d[q] for d in ds]) for q in ("top1", "top5", "median_rank")}
        g1 = " ".join(f"{100 * (d['top1'] - off[d['seed']]['top1']):+.1f}" for d in ds)
        g5 = " ".join(f"{100 * (d['top5'] - off[d['seed']]['top5']):+.1f}" for d in ds)
        print(f"{'banded' if banded else 'fixed':9s} | {avg['top1']:5.1%} | {avg['top5']:5.1%} | "
              f"{avg['median_rank']:6.1f} | {g1:20s} | {g5}")
    if mean_scale:
        print("\nlearned scale per block (geometric mean over all encoders; 1 = keep the hand-picked weight):")
        print("   " + "  ".join(f"{b} x{mean_scale[b]:.2f} (-> {base.blocks[b] * mean_scale[b]:.2f})" for b in blocks))
    print("\nKeep banded=True only if the gains are positive on EVERY split.")


if __name__ == "__main__":
    main()

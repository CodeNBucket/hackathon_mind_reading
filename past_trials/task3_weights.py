"""Task 3: tune the 4 block weights (CLIP, VideoMAE, low-level, AST) and the rhythm-power weight.

Coordinate search on held-out training videos (3 splits): change one weight at a time, keep a
change only if top-5 improves by at least 0.5 points (the noise level is ~1 point, so this is
deliberately conservative), then do a second pass.

    .venv\\Scripts\\python.exe past_trials\\task3_weights.py
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mindreading.model import Config, Model, VideoFeatures, validate  # noqa: E402

BLOCKS = ["clip", "mae", "low", "ast4"]
GRID = {"clip": [0.5, 1.0, 1.5], "mae": [0.3, 0.6, 1.0, 1.5], "low": [0.3, 0.6, 1.0], "ast4": [0.25, 0.5, 0.8]}
POWER_GRID = [0.15, 0.25, 0.35]
MIN_GAIN = 0.005

all_feats = VideoFeatures({b: 1.0 for b in BLOCKS})       # load every block once
seen = {}


def score(weights, power):
    key = (tuple(weights[b] for b in BLOCKS), power)
    if key not in seen:
        cfg = Config(blocks={b: weights[b] for b in ["clip", "ast4", "low", "mae"]}, power_weight=power)
        feats = VideoFeatures.__new__(VideoFeatures)
        feats.blocks, feats.tables = cfg.blocks, all_feats.tables
        t0 = time.time()
        v = validate(cfg, model=Model(cfg, feats))["ensemble"]
        seen[key] = v
        print(f"  weights {dict(zip(BLOCKS, key[0]))} power {power}: top1 {v['top1']:.3f} top5 {v['top5']:.3f} "
              f"median {v['median_rank']:.1f} ({time.time() - t0:.0f} s)", flush=True)
    return seen[key]


best_w = {"clip": 1.0, "mae": 0.6, "low": 0.6, "ast4": 0.5}
best_p = 0.25
best = score(best_w, best_p)
for rnd in (1, 2):
    print(f"--- pass {rnd}", flush=True)
    changed = False
    for b in BLOCKS:
        for val in GRID[b]:
            if val == best_w[b]:
                continue
            w = dict(best_w, **{b: val})
            s = score(w, best_p)
            if s["top5"] >= best["top5"] + MIN_GAIN and s["top1"] >= best["top1"] - 0.003:
                best_w, best, changed = w, s, True
    for p in POWER_GRID:
        if p != best_p:
            s = score(best_w, p)
            if s["top5"] >= best["top5"] + MIN_GAIN and s["top1"] >= best["top1"] - 0.003:
                best_p, best, changed = p, s, True
    if not changed:
        break
print(f"\nBEST weights {best_w} power {best_p}: top1 {best['top1']:.3f} top5 {best['top5']:.3f} "
      f"median {best['median_rank']:.1f}", flush=True)
out = Path(__file__).resolve().parent / "task3_weights_result.json"
out.write_text(json.dumps({"best_weights": best_w, "best_power": best_p, "best": best,
                           "all": [{"weights": dict(zip(BLOCKS, k[0])), "power": k[1], **v} for k, v in seen.items()]},
                          indent=2))

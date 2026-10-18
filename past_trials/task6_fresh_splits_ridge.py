"""Confirm the weight search on 5 NEW random splits (seeds 3-7) that the search never used."""
import sys
import time

sys.path.insert(0, sys.argv[1])
from mindreading.model import Config, Model, VideoFeatures, validate

SEEDS = (3, 4, 5, 6, 7)
base = VideoFeatures({"clip": 1.0, "ast4": 1.0, "low": 1.0, "mae": 1.0})     # load blocks once
runs = [
    ("start: VideoMAE 0.6, power 0.25", {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 0.6}, 0.25),
    ("VideoMAE 1.0", {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 1.0}, 0.25),
    ("chosen: VideoMAE 1.5", {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 1.5}, 0.25),
    ("VideoMAE 2.0", {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 2.0}, 0.25),
    ("VideoMAE 1.5, power 0.35", {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 1.5}, 0.35),
]
for tag, blocks, power in runs:
    cfg = Config(blocks=blocks, power_weight=power, contrastive_weight=0.0)
    feats = VideoFeatures.__new__(VideoFeatures)
    feats.blocks, feats.tables = blocks, base.tables
    t0 = time.time()
    v = validate(cfg, seeds=SEEDS, model=Model(cfg, feats))["ensemble"]
    print(f"{tag:34s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  median {v['median_rank']:.1f}  ({time.time() - t0:.0f} s)",
          flush=True)

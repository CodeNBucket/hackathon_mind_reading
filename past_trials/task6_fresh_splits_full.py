"""Full model (ridge + contrastive) on 5 NEW splits: VideoMAE weight 0.6 / 1.0 / 1.5.

    .venv\Scripts\python.exe past_trials\task6_fresh_splits_full.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mindreading.model import Config, Model, VideoFeatures, validate

SEEDS = (3, 4, 5, 6, 7)
base = VideoFeatures({"clip": 1.0, "ast4": 1.0, "low": 1.0, "mae": 1.0})
for mae in (0.6, 1.0, 1.5):
    blocks = {"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": mae}
    cfg = Config(blocks=blocks, colnorm=False)
    feats = VideoFeatures.__new__(VideoFeatures)
    feats.blocks, feats.tables = blocks, base.tables
    t0 = time.time()
    v = validate(cfg, seeds=SEEDS, model=Model(cfg, feats))
    e, r, c = v["ensemble"], v["wave"], v["wave_c"]
    print(f"FULL VideoMAE {mae}: ensemble top1 {e['top1']:.3f} top5 {e['top5']:.3f} median {e['median_rank']:.1f} | "
          f"ridge-wave top5 {r['top5']:.3f} | contrastive-wave top5 {c['top5']:.3f}  ({time.time() - t0:.0f} s)", flush=True)

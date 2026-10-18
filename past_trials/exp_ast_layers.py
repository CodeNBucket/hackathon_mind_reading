"""Which AST (audio model) layer predicts the EEG best: middle (4), later (8), last (12) or all?

    python past_trials\\exp_ast_layers.py              # audio features ALONE (the clean test)
    python past_trials\\exp_ast_layers.py --combined   # also inside the model with CLIP/VideoMAE/low-level

Question for the "brain vs AI" analysis: does scalp EEG follow the audio model's middle
processing (sound texture, rhythm) more than its last layers (abstract sound categories)?
cache/ast.npz already holds layers 4, 8 and 12 (mean-pooled, 768 each), so nothing is re-extracted.
Each variant: PCA 128 as today. Ridge + rhythm power + column normalisation, contrastive network
off (faster; the question is about the features, not the encoder). validate() protocol, splits 0-2.
-> results/ast_layers.json
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
from mindreading.config import RESULTS_DIR, check_data_dir  # noqa: E402

VARIANTS = {                                   # name -> selection on the (videos, 3 layers, 768) array
    "layer 4 (current)": lambda a: a[:, 0],
    "layer 8": lambda a: a[:, 1],
    "layer 12 (last)": lambda a: a[:, 2],
    "layers 4+8+12": lambda a: a.reshape(len(a), -1),
}
ORIG = M.BLOCKS["ast4"]


def run(base, name, select, seeds, eeg):
    file, key, _, n_pca = ORIG
    M.BLOCKS["ast4"] = (file, key, select, n_pca)
    feats = M.VideoFeatures(base.blocks)
    out = []
    for seed in seeds:
        t0 = time.time()
        model = M.Model(base, feats)
        model.eeg = eeg                                   # share preprocessed EEG
        r = M.validate(base, seeds=(seed,), model=model)["ensemble"]
        out.append({"variant": name, "seed": seed, **r})
        print(f"  {name:18s} split {seed}: top-1 {r['top1']:5.1%}  top-5 {r['top5']:5.1%}  "
              f"median {r['median_rank']:5.1f}  ({time.time() - t0:.0f} s)", flush=True)
    return out


def table(rows, title):
    print(f"\n{title}\nvariant            | top-1 | top-5 | median | top-5 vs layer 4 per split")
    base = {d["seed"]: d for d in rows if d["variant"] == "layer 4 (current)"}
    for name in VARIANTS:
        ds = [d for d in rows if d["variant"] == name]
        avg = {q: np.mean([d[q] for d in ds]) for q in ("top1", "top5", "median_rank")}
        g = " ".join(f"{100 * (d['top5'] - base[d['seed']]['top5']):+.1f}" for d in ds)
        print(f"{name:18s} | {avg['top1']:5.1%} | {avg['top5']:5.1%} | {avg['median_rank']:6.1f} | {g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", action="store_true", help="also test inside the full feature set")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    check_data_dir()
    seeds = [int(x) for x in a.seeds.split(",")]
    eeg, results = {}, {}

    settings = {"audio alone": replace(M.Config(), blocks={"ast4": 1.0}, contrastive_weight=0.0)}
    if a.combined:
        settings["with CLIP, VideoMAE, low-level"] = replace(M.Config(), contrastive_weight=0.0)
    try:
        for label, cfg in settings.items():
            print(f"\n== {label} ==", flush=True)
            results[label] = [r for name, sel in VARIANTS.items() for r in run(cfg, name, sel, seeds, eeg)]
    finally:
        M.BLOCKS["ast4"] = ORIG

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ast_layers.json").write_text(json.dumps(results, indent=1))
    for label, rows in results.items():
        table(rows, f"AST layers, {label} (splits {a.seeds}):")
    print("\nA layer is really better only if it wins on EVERY split.")


if __name__ == "__main__":
    main()

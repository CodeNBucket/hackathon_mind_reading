"""Sweep the channel-whitening shrinkage (Config.whiten) with the final model.

    .venv\\Scripts\\python.exe past_trials\\exp_whiten.py
    .venv\\Scripts\\python.exe past_trials\\exp_whiten.py 0.2 0.3 0.4     # custom values

Everything else stays at the final settings (features, 0-1 s window, 3-member ensemble).
Scores come from held-out TRAINING videos (validate(): 15% hidden, 3 random splits averaged),
never from the sub-01..04 test sets, so the choice stays within the rules.

Background (measured on sub-01 / sub-05 training EEG): raw electrodes correlate 0.78 / 0.83 on
average and one shared component holds 80-84% of the variance. Shrinkage 0 removes all
correlation but blows up the tiniest (noise) directions; 0.3 leaves 0.12 / 0.18.
Round 4 of experiment_log.md found a flat optimum at 0.2-0.4 with the older feature set.

Results are printed and saved to results/whiten_sweep.json.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mindreading.config import CACHE_DIR, RESULTS_DIR, check_data_dir  # noqa: E402
from mindreading.model import Config, Model, VideoFeatures, validate  # noqa: E402

DEFAULT_VALUES = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


def main(values):
    check_data_dir()
    needed = ["clip_b32", "videomae", "lowlevel", "ast"]
    missing = [n for n in needed if not (CACHE_DIR / f"{n}.npz").exists()]
    if missing:
        sys.exit(f"Missing video features in {CACHE_DIR}: {missing}. "
                 "Run notebooks/03_video_features (or run_all.py) first, or copy cache/ from a teammate.")

    feats = VideoFeatures(Config().blocks)   # load the video features once, reuse for every value
    rows = []
    print(f"{'whiten':>7} | {'top-1':>6} | {'top-5':>6} | {'median rank':>11} | time")
    print("-" * 50)
    for w in values:
        t0 = time.time()
        cfg = Config(whiten=w)
        res = validate(cfg, model=Model(cfg, feats))["ensemble"]   # new Model: EEG re-preprocessed with this w
        rows.append({"whiten": w, **res})
        mark = "  <- current" if abs(w - 0.3) < 1e-9 else ""
        print(f"{w:>7.2f} | {res['top1']:>6.1%} | {res['top5']:>6.1%} | {res['median_rank']:>11.1f} | "
              f"{time.time() - t0:.0f} s{mark}", flush=True)

    best = max(rows, key=lambda r: r["top5"])
    ref = next((r for r in rows if abs(r["whiten"] - 0.3) < 1e-9), None)
    print(f"\nbest top-5: whiten={best['whiten']} ({best['top5']:.1%})")
    if ref:
        gain = best["top5"] - ref["top5"]
        print(f"vs current 0.3: {gain:+.1%} top-5. "
              + ("Within noise (< ~0.5 points): keep 0.3." if gain < 0.005 else
                 "Worth checking with more splits before changing Config.whiten."))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "whiten_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    vals = tuple(float(v) for v in sys.argv[1:]) or DEFAULT_VALUES
    main(vals)

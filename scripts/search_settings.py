"""Compare a small set of settings on held-out *training* videos (no test data used).

    .venv\\Scripts\\python.exe scripts\\search_settings.py

Writes results/settings_search.json. Settings are compared on the mean top-5 of the
"combined" score (and each method separately) over all 6 subjects.
"""
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindreading.config import RESULTS_DIR  # noqa: E402
from mindreading.pipeline import Config, summarise, validate_on_train_split  # noqa: E402
from mindreading.video_features import extract_clip  # noqa: E402

BASE = Config()
CANDIDATES = {
    "window 0-1s, pooled":        BASE,
    "window 0-0.6s":              replace(BASE, tmax=0.6),
    "window 0-2s":                replace(BASE, tmax=2.0),
    "window 0-3s":                replace(BASE, tmax=3.0),
    "per-trial normalisation":    replace(BASE, norm="trial"),
    "per-subject models":         replace(BASE, pool_subjects=False),
    "8 frames instead of mean":   replace(BASE, feature_key="frames"),
}


def main():
    feats = extract_clip()
    out = {}
    for name, cfg in CANDIDATES.items():
        t0 = time.time()
        s = summarise(validate_on_train_split(cfg, feats))
        out[name] = {"config": asdict(cfg), "scores": s, "seconds": round(time.time() - t0, 1)}
        line = "  ".join(f"{m}: top1 {v['top1']:.3f} top5 {v['top5']:.3f}" for m, v in s.items())
        print(f"{name:28s} {line}  ({out[name]['seconds']} s)", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "settings_search.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

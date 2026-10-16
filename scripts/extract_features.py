"""Compute and cache every stimulus feature set (GPU used if available).

    .venv\\Scripts\\python.exe scripts\\extract_features.py [names...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mindreading.features import EXTRACTORS  # noqa: E402
from mindreading.video_features import extract_clip  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or ["lowlevel", "clip_b32", "clip_l14", "videomae", "ast"]
    for n in names:
        t0 = time.time()
        r = extract_clip() if n == "clip_b32" else EXTRACTORS[n]()
        print(f"{n}: {', '.join(f'{k} {v.shape}' for k, v in r.items())} ({time.time() - t0:.0f} s)", flush=True)

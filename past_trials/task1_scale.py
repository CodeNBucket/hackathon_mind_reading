"""Task 1: are train and test files on different scales, and does our normalisation handle it?

    .venv\\Scripts\\python.exe past_trials\\task1_scale.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mindreading.config import SUBJECTS  # noqa: E402
from mindreading.data import load_test, load_train  # noqa: E402
from mindreading.preprocess import downsample, prepare  # noqa: E402

print(f"{'subject':8s} {'raw std train':>13s} {'raw std test':>12s} {'ratio':>6s} | "
      f"{'after prepare: train std':>24s} {'test std':>8s} | {'channel-wise std range test (after)':>36s}")
for s in SUBJECTS:
    _, tr = load_train(s)
    te, _ = load_test(s)
    a, b = prepare(tr, whiten=0.3), prepare(te, whiten=0.3)
    ch = b.std(axis=(0, 2))
    print(f"sub-0{s}   {tr.std():13.3f} {te.std():12.3f} {te.std() / tr.std():6.2f} | "
          f"{a.std():24.3f} {b.std():8.3f} | {ch.min():.3f} - {ch.max():.3f}")

# Is the scale difference the same over time and channels (a plain gain) or does the shape differ?
_, tr = load_train(1)
te, _ = load_test(1)
r_t = downsample(te).std(axis=(0, 1)) / downsample(tr).std(axis=(0, 1))
r_c = downsample(te).std(axis=(0, 2)) / downsample(tr).std(axis=(0, 2))
print(f"\nsub-01 test/train std ratio over time: min {r_t.min():.2f}, max {r_t.max():.2f}; "
      f"over channels: min {r_c.min():.2f}, max {r_c.max():.2f}")

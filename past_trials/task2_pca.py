"""Task 2: dimensionality reduction of the VideoMAE and AST blocks (final model, with rhythm power).

    .venv\\Scripts\\python.exe past_trials\\task2_pca.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mindreading.model as M  # noqa: E402
from mindreading.model import Config, validate  # noqa: E402

ORIG = dict(M.BLOCKS)
ORIG_LOAD = M.load_block


def load_block_rp(name, dims=128):
    """Like load_block, but a random projection instead of PCA (for the blocks that use PCA)."""
    from sklearn.preprocessing import StandardScaler
    file, key, select, n_pca = M.BLOCKS[name]
    if not n_pca:
        return ORIG_LOAD(name)
    z = np.load(M.CACHE_DIR / f"{file}.npz")
    arr = z[key].astype(np.float32)
    arr = select(arr) if select is not None else arr
    X = arr.reshape(len(arr), -1)
    ids = z["ids"].astype(int)
    tr = ids <= 1000
    X = StandardScaler().fit(X[tr]).transform(X)
    X = GaussianRandomProjection(dims, random_state=0).fit(X[tr]).transform(X)
    X = (X - X[tr].mean(0)) / X[tr].std(0).mean()
    X /= np.sqrt(X.shape[1])
    return dict(zip(ids.tolist(), X.astype(np.float32)))


def with_dims(mae, ast):
    b = dict(ORIG)
    b["mae"] = ("videomae", "layers", None, mae)
    b["ast4"] = ("ast", "layers", lambda a: a[:, 0], ast)
    return b


runs = [("PCA 128 / 128 (current)", with_dims(128, 128), ORIG_LOAD),
        ("PCA 64 / 64", with_dims(64, 64), ORIG_LOAD),
        ("PCA 256 / 256", with_dims(256, 256), ORIG_LOAD),
        ("PCA 512 / 512", with_dims(512, 512), ORIG_LOAD),
        ("no reduction (2304 / 768)", with_dims(None, None), ORIG_LOAD),
        ("random projection 128 / 128", with_dims(128, 128), load_block_rp)]
for tag, blocks, loader in runs:
    M.BLOCKS.clear(); M.BLOCKS.update(blocks); M.load_block = loader
    t0 = time.time()
    v = validate(Config())["ensemble"]
    print(f"{tag:30s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  median rank {v['median_rank']:.1f}"
          f"  ({time.time() - t0:.0f} s)", flush=True)

"""Combinations of feature blocks under the best EEG settings (validated on held-out training videos).

    .venv\\Scripts\\python.exe past_trials\\exp_combo.py <name>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_eeg import CACHE_DIR, concat, evaluate, load_block  # noqa: E402

PREP = dict(tmin=0.0, tmax=1.0, car=False, whiten=0.3)
ENS = dict(mode="both", use_ga=True)


def blocks():
    avail = {p.stem for p in CACHE_DIR.glob("*.npz")}
    b = {"clip": load_block("clip_b32", "mean")}
    if "ast" in avail:
        b["ast4"] = load_block("ast", "layers", lambda a: a[:, 0], n_pca=128)
        b["ast"] = load_block("ast", "layers", lambda a: a.reshape(len(a), -1), n_pca=128)
    if "lowlevel" in avail:
        b["low"] = load_block("lowlevel", "timecourse", lambda a: a[:, :12])
    if "videomae" in avail:
        b["mae"] = load_block("videomae", "layers", lambda a: a.reshape(len(a), -1), n_pca=128)
    if "clip_l14" in avail:
        b["l14"] = load_block("clip_l14", "embed", lambda a: a.mean(1))
        b["l14cls"] = load_block("clip_l14", "cls", lambda a: a.mean(1).reshape(len(a), -1), n_pca=256)
    return b


if __name__ == "__main__":
    B = blocks()
    if sys.argv[1] == "l14":
        import exp_eeg
        exp_eeg.PER_TARGET = True
        best = [B["clip"], B["ast4"], B["low"], B["mae"]]
        wb = [1, 0.5, 0.6, 0.6]
        evaluate(concat(*best, weights=wb), PREP, **ENS, tag="current best (reference)")
        evaluate(concat(B["l14"]), PREP, **ENS, tag="CLIP-L14 embed alone")
        evaluate(concat(B["l14cls"]), PREP, **ENS, tag="CLIP-L14 layers 6/12/18/24 alone")
        evaluate(concat(B["l14"], B["ast4"], B["low"], B["mae"], weights=wb), PREP, **ENS, tag="swap CLIP-B32 -> CLIP-L14")
        for w in (0.4, 0.8):
            evaluate(concat(*best, B["l14"], weights=wb + [w]), PREP, **ENS, tag=f"best + CLIP-L14 embed ({w})")
            evaluate(concat(*best, B["l14cls"], weights=wb + [w]), PREP, **ENS, tag=f"best + CLIP-L14 layers ({w})")
        sys.exit()
    if sys.argv[1] == "mae2":
        import exp_eeg
        exp_eeg.PER_TARGET = True
        base = [B["clip"], B["ast4"], B["low"]]
        variants = {f"videomae layer {l}": load_block("videomae", "layers", lambda a, i=i: a[:, i], n_pca=128)
                    for i, l in enumerate((4, 8, 12))}
        variants["videomae all layers, PCA 256"] = load_block("videomae", "layers", lambda a: a.reshape(len(a), -1), n_pca=256)
        variants["videomae all layers, no PCA"] = load_block("videomae", "layers", lambda a: a.reshape(len(a), -1))
        for tag, b in variants.items():
            evaluate(concat(*base, b, weights=[1, 0.5, 0.6, 0.6]), PREP, **ENS, tag=f"clip+ast4+low + {tag}")
        sys.exit()
    if sys.argv[1] == "mae":
        import exp_eeg
        exp_eeg.PER_TARGET = True
        base = [B["clip"], B["ast4"], B["low"]]
        evaluate(concat(B["mae"]), PREP, **ENS, tag="videomae alone")
        for w in (0.3, 0.6, 1.0):
            evaluate(concat(*base, B["mae"], weights=[1, 0.5, 0.6, w]), PREP, **ENS, tag=f"clip+ast4+low + videomae ({w})")
        sys.exit()
    if sys.argv[1] == "alpha":
        import exp_eeg
        f = concat(B["clip"], B["ast4"], B["low"], weights=[1, 0.5, 0.6])
        exp_eeg.PER_TARGET = True
        evaluate(f, PREP, **ENS, tag="ridge strength per EEG feature")
        sys.exit()
    if sys.argv[1] == "low":
        evaluate(concat(B["clip"], B["ast4"], weights=[1, 0.5]), PREP, **ENS, tag="clip + ast4 (reference)")
        evaluate(concat(B["low"]), PREP, **ENS, tag="lowlevel alone")
        for w in (0.3, 0.6, 1.0):
            evaluate(concat(B["clip"], B["ast4"], B["low"], weights=[1, 0.5, w]), PREP, **ENS,
                     tag=f"clip + ast4 + lowlevel ({w})")
        sys.exit()
    if sys.argv[1] == "whiten2":
        f = concat(B["clip"], B["ast4"], weights=[1, 0.5])
        for sh in (0.01, 0.05, 0.1, 0.4):
            evaluate(f, {**PREP, "car": False, "whiten": sh}, **ENS, tag=f"whitening only (shrink {sh})")
        evaluate(f, {**PREP, "car": False, "whiten": 0.1, "tmax": 0.8}, **ENS, tag="whitening 0.1, window 0-0.8")
        evaluate(f, {**PREP, "car": False, "whiten": 0.1, "tmax": 1.2}, **ENS, tag="whitening 0.1, window 0-1.2")
        sys.exit()
    if sys.argv[1] == "whiten":
        f = concat(B["clip"], B["ast4"], weights=[1, 0.5])
        evaluate(f, PREP, **ENS, tag="CAR (reference)")
        for sh in (0.05, 0.2, 0.5):
            evaluate(f, {**PREP, "whiten": sh}, **ENS, tag=f"CAR + whitening (shrink {sh})")
        evaluate(f, {**PREP, "car": False, "whiten": 0.2}, **ENS, tag="whitening only (shrink 0.2)")
        sys.exit()
    combos = {
        "audio": [("clip alone", ["clip"], None),
                  ("ast4 alone", ["ast4"], None),
                  ("clip + ast4 (0.3)", ["clip", "ast4"], [1, 0.3]),
                  ("clip + ast4 (0.5)", ["clip", "ast4"], [1, 0.5]),
                  ("clip + ast4 (1.0)", ["clip", "ast4"], [1, 1]),
                  ("clip + ast all layers (0.5)", ["clip", "ast"], [1, 0.5])],
        "all": [(f"{' + '.join(k)}", k, None) for k in (
            ["clip", "low"], ["clip", "mae"], ["clip", "l14"], ["l14", "l14cls"],
            ["clip", "ast4", "low"], ["clip", "ast4", "mae"], ["clip", "l14", "ast4"],
            ["clip", "l14", "ast4", "low", "mae"], ["clip", "l14", "l14cls", "ast4", "low", "mae"])
            if all(x in B for x in k)],
    }[sys.argv[1]]
    for tag, keys, w in combos:
        evaluate(concat(*[B[k] for k in keys], weights=w), PREP, **ENS, tag=tag)

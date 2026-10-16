"""Look inside the organisers' EEG files without a notebook.

    python peek_data.py                       # sub-01, first training video
    python peek_data.py --sub 5 --video 35    # sub-05, training video 0035.mp4

Prints what is in trn/, tst/ and tst_stim/ for one subject, and writes to out/:
  - recording_sub-XX_vid-N.png : the 17 electrodes over 3 s for one training video
  - recording_sub-XX_vid-N.csv : the same numbers as a table (opens in Excel)
  - average_sub-XX.png         : the average over all training videos (the "ERP")
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_DATA = r"C:\Users\canbe\Desktop\eeg-mind-reading-challenge\eeg-mind-reading-challenge"
SFREQ = 500  # Hz: 1500 samples = 3 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--video", type=int, default=None, help="training video id, e.g. 35 for 0035.mp4")
    args = ap.parse_args()
    data, sub = Path(args.data), f"sub-{args.sub:02d}"
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)

    # trn/: a .npz is a zip of named numpy arrays. Here: name = video id, array = 17 x 1500.
    trn = np.load(data / "trn" / f"{sub}_trn.npz")
    ids = sorted(trn.files, key=int)
    vid = str(args.video) if args.video is not None else ids[0]
    rec = trn[vid]
    print(f"== {sub} training file: {len(ids)} recordings")
    print(f"   video ids (= names inside the file): {ids[:8]} ... {ids[-3:]}")
    print(f"   one recording, video {vid} ({int(vid):04d}.mp4): shape {rec.shape} = 17 electrodes x 1500 time points")
    print(f"   first 5 values of electrode 1: {np.round(rec[0, :5], 2)}")

    # tst/: 102 recordings, shuffled; tst_stim/: which video each one was (sub-01..04 only).
    tst = np.load(data / "tst" / f"{sub}_tst.npy")
    print(f"== {sub} test file: shape {tst.shape} (102 recordings, order shuffled)")
    stim = data / "tst_stim" / f"{sub}_tst_stim.npy"
    if stim.exists():
        print(f"   answers: first 5 test recordings are videos {np.load(stim)[:5]}")
    else:
        print("   answers: none (held-out subject, graded by the organisers)")

    t = np.arange(rec.shape[1]) / SFREQ
    np.savetxt(out / f"recording_{sub}_vid-{vid}.csv",
               np.column_stack([t, rec.T]), delimiter=",", fmt="%.4f",
               header="time_s," + ",".join(f"electrode_{i + 1}" for i in range(rec.shape[0])), comments="")

    gap = 4 * rec.std()  # vertical spacing so the 17 lines don't overlap
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(rec.shape[0]):
        ax.plot(t, rec[i] + i * gap, lw=0.6)
    ax.set(xlabel="seconds after video start", ylabel="electrode (stacked)",
           title=f"{sub}, one recording: video {int(vid):04d}.mp4")
    ax.set_yticks(np.arange(rec.shape[0]) * gap, [f"e{i + 1}" for i in range(rec.shape[0])])
    fig.tight_layout()
    fig.savefig(out / f"recording_{sub}_vid-{vid}.png", dpi=110)

    erp = np.mean([trn[k] for k in ids], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, erp.T, lw=0.8)
    ax.set(xlabel="seconds after video start", ylabel="average signal",
           title=f"{sub}: average over all {len(ids)} training videos (one line per electrode)")
    fig.tight_layout()
    fig.savefig(out / f"average_{sub}.png", dpi=110)
    print(f"== wrote {out}")


if __name__ == "__main__":
    main()

"""Generate a SYNTHETIC EEG dataset shaped like a typical video-viewing experiment.

Nothing here is real brain data. The signals are built from textbook ERP shapes so you can
practise loading, plotting and decoding before the real data arrives.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
OUT = Path(__file__).parent

SFREQ = 100                     # samples per second (already downsampled)
TMIN, TMAX = -0.2, 3.0          # epoch: 200 ms before video onset to 3 s after
times = np.arange(int(round((TMAX - TMIN) * SFREQ))) / SFREQ + TMIN
CH_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
            "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
N_CH = len(CH_NAMES)
N_SUBJECTS, N_VIDEOS, N_REPS = 4, 20, 5

# --- the stimuli: 20 three-second videos with a few known properties -------------------
categories = ["face", "animal", "scene", "object"]
videos = pd.DataFrame({
    "video_id": [f"vid_{i:02d}" for i in range(N_VIDEOS)],
    "category": [categories[i % 4] for i in range(N_VIDEOS)],
    "mean_luminance": rng.uniform(0.2, 0.8, N_VIDEOS).round(3),   # 0 = black, 1 = white
    "motion_energy": rng.uniform(0.0, 1.0, N_VIDEOS).round(3),    # 0 = static, 1 = lots of motion
    "cut_time_s": rng.choice([np.nan, 1.0, 1.5, 2.0], N_VIDEOS),  # a scene cut mid-video, or none
})

# each video's moment-to-moment brightness/motion fluctuation (smooth random curve, ~2-4 Hz)
_k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2); _k /= _k.sum()
fluct = np.stack([np.convolve(rng.normal(size=times.size), _k, mode="same") for _ in range(N_VIDEOS)])
fluct = fluct / fluct.std(axis=1, keepdims=True) * (times > 0.1)

def gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)

def scalp_weights(kind):
    w = np.zeros(N_CH)
    for i, ch in enumerate(CH_NAMES):
        if kind == "occipital":
            w[i] = 1.0 if ch.startswith("O") else 0.6 if ch.startswith("PO") else 0.25 if ch.startswith("P") else 0.05
        elif kind == "occtemp":       # face-sensitive sites, right-lateralized
            w[i] = {"P8": 1.0, "PO8": 0.9, "P7": 0.7, "PO7": 0.6, "O2": 0.4, "O1": 0.3}.get(ch, 0.03)
        elif kind == "parietal":
            w[i] = 1.0 if ch in ("Pz", "POz", "P3", "P4") else 0.5 if ch.startswith("PO") or ch.startswith("CP") else 0.05
        elif kind == "frontal":
            w[i] = 1.0 if ch in ("Fp1", "Fp2") else 0.4 if ch.startswith("F") else 0.0
    return w

W_OCC, W_FACE, W_PAR, W_EYE = (scalp_weights(k) for k in ("occipital", "occtemp", "parietal", "frontal"))

def video_response(v):
    """Noise-free brain response (channels x time) to one video, in microvolts."""
    x = np.zeros((N_CH, times.size))
    lum, mot = v.mean_luminance, v.motion_energy
    # P1 ~100 ms over occipital channels, bigger for brighter/higher-contrast clips
    x += np.outer(W_OCC, (3 + 6 * lum) * gauss(times, 0.10, 0.02))
    # N1 ~170 ms; the face-sensitive N170 is extra large for faces at occipito-temporal sites
    x -= np.outer(W_OCC, 3 * gauss(times, 0.17, 0.025))
    if v.category == "face":
        x -= np.outer(W_FACE, 6 * gauss(times, 0.17, 0.025))
    # motion-onset negativity ~200 ms over parietal sites, scales with motion
    x -= np.outer(W_PAR, 5 * mot * gauss(times, 0.21, 0.03))
    # late positive component ~400 ms, differs a bit by category (meaning)
    x += np.outer(W_PAR, (2 + categories.index(v.category)) * gauss(times, 0.40, 0.08))
    # sustained response while the video plays, plus a small re-onset at a scene cut
    x += np.outer(W_OCC, 1.0 * lum * (times > 0.3) * (times < 3.0))
    # the brain tracks each video's own brightness/motion changes over time (~100 ms lag)
    x += np.outer(W_OCC, 2.0 * fluct[v.Index])
    if not np.isnan(v.cut_time_s):
        x += np.outer(W_OCC, 4 * gauss(times, v.cut_time_s + 0.10, 0.02))
    return x

def pink_noise(n_ch, n_t, scale):
    f = np.fft.rfftfreq(n_t, 1 / SFREQ); f[0] = 1
    spec = (rng.normal(size=(n_ch, f.size)) + 1j * rng.normal(size=(n_ch, f.size))) / np.sqrt(f)
    sig = np.fft.irfft(spec, n=n_t)
    return scale * sig / sig.std()

templates = np.stack([video_response(v) for v in videos.itertuples()])
X, rows = [], []
for s in range(N_SUBJECTS):
    gain = rng.uniform(0.6, 1.4)                         # people differ in signal strength
    mix = np.eye(N_CH) + 0.08 * rng.normal(size=(N_CH, N_CH))   # ...and in how sources map to the scalp
    alpha_freq = rng.uniform(9, 11)
    order = [(vid, rep) for vid in range(N_VIDEOS) for rep in range(N_REPS)]
    rng.shuffle(order)                                   # randomized presentation order
    for trial, (vid, rep) in enumerate(order):
        sig = gain * mix @ templates[vid]
        noise = pink_noise(N_CH, times.size, 8.0)        # background EEG dominates single trials
        alpha = np.outer(W_OCC, 4 * np.sin(2 * np.pi * alpha_freq * times + rng.uniform(0, 6.3)))
        alpha *= np.where(times > 0.3, 0.4, 1.0)         # alpha drops when the video engages vision
        blink = np.zeros_like(sig)
        if rng.random() < 0.2:                           # 20% of trials contain an eye blink
            t0 = rng.uniform(0.2, 2.8)
            blink = np.outer(W_EYE, 80 * gauss(times, t0, 0.06))
        X.append((sig + noise + alpha + blink).astype(np.float32))
        rows.append({"subject": f"sub-{s + 1:02d}", "trial": trial, "video_id": videos.video_id[vid],
                     "repetition": rep + 1, "session": 1 + trial // 50, "has_blink": bool(blink.any())})

X = np.stack(X)                                          # (trials, channels, time)
meta = pd.DataFrame(rows)
np.save(OUT / "eeg_epochs.npy", X)
meta.to_csv(OUT / "trials.csv", index=False)
videos.to_csv(OUT / "videos.csv", index=False)

# a toy 16-number "embedding" per video, standing in for a video foundation model's output:
# what is in the clip (category), how it looks (brightness, motion) and how it unfolds over time
dyn = fluct[:, (times > 0.1) & (times < 2.5)]
dyn = dyn[:, : dyn.shape[1] // 29 * 29].reshape(N_VIDEOS, 8, -1).mean(-1)   # 8 coarse time steps
emb = np.column_stack([videos.mean_luminance, videos.motion_energy,
                       *(videos.category == c for c in categories),
                       ~np.isnan(videos.cut_time_s), rng.normal(0, 0.1, N_VIDEOS), dyn]).astype(np.float32)
np.save(OUT / "video_embeddings_toy.npy", emb)

json.dump({"sfreq_hz": SFREQ, "tmin_s": TMIN, "tmax_s": TMAX, "unit": "microvolt",
           "array_shape": "trials x channels x time", "ch_names": CH_NAMES,
           "reference": "average", "filter": "none applied (practise it yourself)",
           "note": "SYNTHETIC data for practice only"}, open(OUT / "info.json", "w"), indent=2)
print(X.shape, meta.shape)

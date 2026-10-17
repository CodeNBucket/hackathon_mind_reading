"""EEG preprocessing.

1. Downsample 500 Hz -> 100 Hz with an anti-aliasing FIR filter (as the organisers do).
2. Normalise. The test files are scaled differently from the training files
   (std ~0.32 vs ~0.49 for sub-01), so every set is normalised with its own statistics.
   This is unsupervised and explicitly allowed for test recordings.
3. Cut a time window (seconds after video onset).
"""
import numpy as np
from scipy.signal import decimate

DEC = 5          # 500 Hz / 5 = 100 Hz
SFREQ_OUT = 100


def downsample(eeg):
    """(n, 17, 1500) -> (n, 17, 300)."""
    return decimate(eeg, DEC, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)


def normalise(eeg, mode="set"):
    """mode='set': z-score each channel over all recordings and time points of this set.
    mode='trial': z-score each recording's channels separately (removes per-recording offsets)."""
    if mode == "set":
        mu = eeg.mean(axis=(0, 2), keepdims=True)
        sd = eeg.std(axis=(0, 2), keepdims=True)
    elif mode == "trial":
        mu = eeg.mean(axis=2, keepdims=True)
        sd = eeg.std(axis=2, keepdims=True)
    else:
        raise ValueError(mode)
    return (eeg - mu) / (sd + 1e-8)


def window(eeg, tmin=0.0, tmax=1.0):
    a, b = int(round(tmin * SFREQ_OUT)), int(round(tmax * SFREQ_OUT))
    return eeg[:, :, a:b]


def common_average(eeg):
    """Subtract the mean over electrodes at every time point (common average reference).
    The 17 channels are ~0.77 correlated; the shared part is mostly noise for our purpose."""
    return eeg - eeg.mean(axis=1, keepdims=True)


def whiten_channels(eeg, shrink=0.3):
    """Decorrelate the electrodes using this file's own channel covariance, shrunk towards
    the identity (multivariate noise normalisation). Unsupervised, so allowed on test files."""
    flat = eeg.transpose(1, 0, 2).reshape(eeg.shape[1], -1)
    C = np.cov(flat)
    C = (1 - shrink) * C + shrink * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T
    return np.einsum("ij,njt->nit", W, eeg).astype(np.float32)


POWER_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))   # delta, theta, alpha, beta, low gamma (Hz)


def band_power(eeg, whiten=0.3, bands=POWER_BANDS, tmin=0.0, tmax=3.0, bin_s=0.5):
    """How strong each brain rhythm is: raw (n, 17, 1500) -> (n, 17 x bands x bins).
    Log power per electrode, frequency band and 0.5-s bin over 0-3 s, z-scored within this
    file (unsupervised, allowed on test files). Captures the slower 'state' of the brain
    (e.g. alpha drops with attention / motion) that the exact wave shape after 1 s doesn't."""
    from scipy.signal import butter, sosfiltfilt
    x = normalise(whiten_channels(downsample(eeg), whiten) if whiten else downsample(eeg))
    a, b, w = int(round(tmin * SFREQ_OUT)), int(round(tmax * SFREQ_OUT)), int(round(bin_s * SFREQ_OUT))
    out = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype="band", fs=SFREQ_OUT, output="sos")
        p = sosfiltfilt(sos, x, axis=-1)[:, :, a:b] ** 2
        out.append(np.log(p.reshape(len(x), x.shape[1], (b - a) // w, w).mean(-1) + 1e-6))
    f = np.concatenate(out, axis=2).reshape(len(x), -1)
    return ((f - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)


def prepare(eeg, norm="set", tmin=0.0, tmax=1.0, car=False, whiten=0.0):
    """Full preprocessing: raw (n, 17, 1500) -> (n, 17, T) at 100 Hz.
    whiten > 0 enables channel whitening with that shrinkage."""
    x = downsample(eeg)
    if car:
        x = common_average(x)
    if whiten:
        x = whiten_channels(x, whiten)
    return window(normalise(x, norm), tmin, tmax)

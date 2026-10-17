"""Stimulus features for all 1,102 videos, cached in cache/<name>.npz.

Each extractor returns a dict of arrays whose first axis is the video (same order as `ids`).

lowlevel   no model. Time courses at 10 Hz over the 3 s (30 bins): brightness, contrast,
           colour saturation, motion (frame difference), edge energy, scene-cut strength,
           audio loudness (RMS, dB) and audio spectral flux (sound onsets).
videomae   VideoMAE (Kinetics-400 actions) on 16 frames: mean-pooled hidden state of
           layers 4/8/12 (768-d each). Captures motion and actions.
ast        Audio Spectrogram Transformer (AudioSet sound events) on the 3 s soundtrack:
           mean-pooled hidden state of layers 4/8/12 (768-d each).
"""
import time

import av
import numpy as np

from .config import CACHE_DIR, TEST_IDS, TRAIN_IDS
from .data import video_path

ALL_IDS = np.array(TRAIN_IDS + TEST_IDS)

# Exact Hugging Face commits of the public pretrained models we use (reproducibility).
REVISIONS = {
    "clip_b32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",   # openai/clip-vit-base-patch32
    "videomae": "488eb9a0565f257b32866000305c8178965eb9f6",   # MCG-NJU/videomae-base-finetuned-kinetics
    "ast": "f826b80d28226b62986cc218e5cec390b1096902",        # MIT/ast-finetuned-audioset-10-10-0.4593
}
DURATION = 3.0


# ---------------------------------------------------------------- decoding

def read_video(path):
    """All frames as uint8 (T, H, W, 3) and their timestamps in seconds."""
    with av.open(str(path), metadata_errors="ignore") as c:
        frames, times = [], []
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
            times.append(float(f.time) if f.time is not None else len(times) / 30.0)
    return np.stack(frames), np.array(times)


def read_audio(path, sr=16000, seconds=DURATION):
    """Mono float32 audio at `sr`, padded/cut to `seconds`. Silent if the file has no audio."""
    n = int(sr * seconds)
    out = np.zeros(n, np.float32)
    with av.open(str(path), metadata_errors="ignore") as c:
        if not c.streams.audio:
            return out
        res = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for fr in c.decode(audio=0):
            for r in res.resample(fr):
                chunks.append(r.to_ndarray().reshape(-1))
        for r in res.resample(None):
            chunks.append(r.to_ndarray().reshape(-1))
    if chunks:
        a = np.concatenate(chunks)[:n]
        out[:len(a)] = a
    return out


def frames_at(frames, times, t_list):
    idx = [int(np.argmin(np.abs(times - t))) for t in t_list]
    return frames[idx]


# ---------------------------------------------------------------- caching

def cached(name, fn, ids=ALL_IDS, verbose=True):
    """Run fn(video_id) -> dict of arrays for every id, stack, cache. Resumable."""
    out = CACHE_DIR / f"{name}.npz"
    if out.exists():
        z = np.load(out)
        return {k: z[k] for k in z.files}
    CACHE_DIR.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for i, vid in enumerate(ids):
        rows.append(fn(int(vid)))
        if verbose and (i + 1) % 100 == 0:
            print(f"  {name}: {i + 1}/{len(ids)} videos, {time.time() - t0:.0f} s", flush=True)
    res = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    res["ids"] = np.asarray(ids)
    np.savez(out, **res)
    return res


# ---------------------------------------------------------------- low-level

def _bin(x, t, n_bins=30, duration=DURATION):
    edges = np.linspace(0, duration, n_bins + 1)
    return np.array([x[(t >= a) & (t < b)].mean() if ((t >= a) & (t < b)).any() else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])


def _fill(v):
    v = v.copy()
    bad = np.isnan(v)
    if bad.all():
        return np.zeros_like(v)
    v[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), v[~bad])
    return v


def lowlevel_one(vid):
    frames, t = read_video(video_path(vid))
    f = frames.astype(np.float32) / 255.0
    gray = f.mean(-1)
    lum = gray.mean((1, 2))
    contrast = gray.std((1, 2))
    mx, mn = f.max(-1), f.min(-1)
    sat = ((mx - mn) / (mx + 1e-6)).mean((1, 2))
    diff = np.r_[0.0, np.abs(np.diff(gray, axis=0)).mean((1, 2))]
    gx, gy = np.abs(np.diff(gray, axis=2)).mean((1, 2)), np.abs(np.diff(gray, axis=1)).mean((1, 2))
    edges = gx + gy
    hist = np.stack([np.histogram(g, 16, (0, 1))[0] for g in gray]).astype(np.float32)
    hist /= hist.sum(1, keepdims=True)
    cut = np.r_[0.0, 0.5 * np.abs(np.diff(hist, axis=0)).sum(1)]
    vis = np.stack([_fill(_bin(v, t)) for v in (lum, contrast, sat, diff, edges, cut)], 1)

    sr = 16000
    a = read_audio(video_path(vid), sr)
    hop = sr // 100                                   # 10 ms frames
    fr = a[: len(a) // hop * hop].reshape(-1, hop)
    rms_db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-5)
    spec = np.abs(np.fft.rfft(fr * np.hanning(hop), axis=1))
    flux = np.r_[0.0, np.clip(np.diff(np.log(spec + 1e-6), axis=0), 0, None).mean(1)]
    ta = np.arange(len(fr)) / 100
    aud = np.stack([_fill(_bin(rms_db, ta)), _fill(_bin(flux, ta))], 1)
    return {"timecourse": np.concatenate([vis, aud], 1).astype(np.float32)}   # (30 bins, 8)


LOWLEVEL_NAMES = ["brightness", "contrast", "saturation", "motion", "edges", "scene_cut",
                  "loudness_db", "sound_onsets"]


def extract_lowlevel():
    return cached("lowlevel", lowlevel_one)


# ---------------------------------------------------------------- deep models

def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_videomae(layers=(4, 8, 12)):
    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    name, rev = "MCG-NJU/videomae-base-finetuned-kinetics", REVISIONS["videomae"]
    dev = _device()
    model = VideoMAEModel.from_pretrained(name, revision=rev).to(dev).eval()
    proc = VideoMAEImageProcessor.from_pretrained(name, revision=rev)

    @torch.no_grad()
    def one(vid):
        frames, t = read_video(video_path(vid))
        clip = list(frames_at(frames, t, np.linspace(0, DURATION - 0.05, 16)))
        px = proc(clip, return_tensors="pt")["pixel_values"].to(dev)
        hs = model(pixel_values=px, output_hidden_states=True).hidden_states
        return {"layers": torch.stack([hs[l].mean(1)[0] for l in layers]).cpu().numpy().astype(np.float32)}

    return cached("videomae", one)


def extract_ast(layers=(4, 8, 12)):
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    name, rev = "MIT/ast-finetuned-audioset-10-10-0.4593", REVISIONS["ast"]
    dev = _device()
    model = ASTModel.from_pretrained(name, revision=rev).to(dev).eval()
    fe = ASTFeatureExtractor.from_pretrained(name, revision=rev)

    @torch.no_grad()
    def one(vid):
        a = read_audio(video_path(vid), 16000)
        x = fe(a, sampling_rate=16000, return_tensors="pt")["input_values"].to(dev)
        hs = model(input_values=x, output_hidden_states=True).hidden_states
        return {"layers": torch.stack([hs[l].mean(1)[0] for l in layers]).cpu().numpy().astype(np.float32),
                "silent": np.array(float(np.abs(a).max() < 1e-4), np.float32)}

    return cached("ast", one)


EXTRACTORS = {"lowlevel": extract_lowlevel, "videomae": extract_videomae, "ast": extract_ast}

"""Video -> CLIP features ("cards").

For each video we take N_FRAMES evenly spaced frames, embed each with CLIP, and keep
both the per-frame vectors (N_FRAMES x 512) and their normalised mean (512).
Results are cached in cache/<name>.npz so this runs once.
"""
import time
from pathlib import Path

import av
import numpy as np

from .config import CACHE_DIR, TEST_IDS, TRAIN_IDS
from .data import video_path

CLIP_NAME = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
N_FRAMES = 8


def read_frames(path: Path, n: int = N_FRAMES):
    with av.open(str(path), metadata_errors="ignore") as c:  # some files have non-UTF-8 metadata
        frames = list(c.decode(video=0))
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i].to_image() for i in idx]  # convert only the frames we keep


def extract_clip(video_ids=None, name="clip_b32", batch_videos=16, verbose=True):
    """Returns dict(ids [n], frames [n, N_FRAMES, 512], mean [n, 512]); cached."""
    out = CACHE_DIR / f"{name}.npz"
    if out.exists():
        z = np.load(out)
        return {k: z[k] for k in z.files}
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_grad_enabled(False)
    model = CLIPModel.from_pretrained(CLIP_NAME, revision=CLIP_REVISION).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME, revision=CLIP_REVISION)
    ids = np.array(video_ids if video_ids is not None else TRAIN_IDS + TEST_IDS)
    feats = np.zeros((len(ids), N_FRAMES, 512), np.float32)
    CACHE_DIR.mkdir(exist_ok=True)
    partial = CACHE_DIR / f"{name}_partial.npz"
    start = 0
    if partial.exists():  # resume an interrupted run
        z = np.load(partial)
        start = int(z["done"])
        feats[:start] = z["frames"][:start]
    t0 = time.time()
    for s in range(start, len(ids), batch_videos):
        chunk = ids[s:s + batch_videos]
        imgs = [im for vid in chunk for im in read_frames(video_path(int(vid)))]
        emb = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
        emb = getattr(emb, "pooler_output", emb)  # transformers 5 may wrap the tensor
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats[s:s + len(chunk)] = emb.reshape(len(chunk), N_FRAMES, -1).numpy()
        if (s // batch_videos) % 10 == 0:
            np.savez(partial, frames=feats, done=s + len(chunk))
            if verbose:
                print(f"  {s + len(chunk)}/{len(ids)} videos, {time.time() - t0:.0f} s", flush=True)
    mean = feats.mean(1)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    np.savez(out, ids=ids, frames=feats, mean=mean)
    partial.unlink(missing_ok=True)
    return {"ids": ids, "frames": feats, "mean": mean}


def lookup(features: dict, video_ids, key="mean"):
    """Rows of features[key] for the given video ids, in that order."""
    pos = {int(v): i for i, v in enumerate(features["ids"])}
    return features[key][[pos[int(v)] for v in video_ids]]

"""Video -> foundation-model output, in 4 stages.

    python run.py                      # dog clip, seconds 20-23
    python run.py --video my.mp4 --start 0
    python run.py --skip-videomae      # CLIP only

Stages
  1. cut a 3 s clip from the video and read its frames   (pipeline/video.py)
  2. CLIP: 8 frames -> 8 x 512 vectors -> 1 video vector  (pipeline/clip_embed.py)
  3. VideoMAE: 16 frames -> action label + 768 vector     (pipeline/videomae_classify.py)
  4. save everything to outputs/
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Keep downloaded model weights inside this project (models/) instead of ~/.cache.
os.environ.setdefault("HF_HOME", str(HERE / "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pipeline import clip_embed, video, videomae_classify  # noqa: E402

TEXTS = ["a dog running", "a cat sleeping", "a city skyline at night",
         "people dancing", "a car on a highway"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=HERE / "data" / "dog_agility_full.ogv")
    ap.add_argument("--start", type=float, default=20.0, help="seconds into the video")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--clip-frames", type=int, default=8)
    ap.add_argument("--skip-videomae", action="store_true")
    args = ap.parse_args()

    if sys.version_info[:2] != (3, 11):
        print(f"warning: hackathon expects Python 3.11, this is {platform.python_version()}")

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Stage 1
    print("[1/4] cutting clip and reading frames")
    clip_path = HERE / "data" / "clip_3s.mp4"
    video.cut_clip(args.video, clip_path, args.start, args.seconds)
    frames, fps = video.read_frames(clip_path)
    print(f"      {len(frames)} frames at {fps:.2f} fps, {frames[0].size[0]}x{frames[0].size[1]}")

    # Stage 2
    print("[2/4] CLIP embeddings")
    t0 = time.time()
    model, proc = clip_embed.load()
    imgs, idx = video.sample(frames, args.clip_frames)
    fvec, vvec, tvec, scale = clip_embed.embed(model, proc, imgs, TEXTS)
    cos = tvec @ vvec
    probs = (scale * cos).softmax(0)
    per_frame = tvec[0] @ fvec.T
    np.save(out_dir / "clip_frame_vectors.npy", fvec.numpy())   # (8, 512)
    np.save(out_dir / "clip_video_vector.npy", vvec.numpy())    # (512,)
    clip_res = {
        "model": clip_embed.MODEL_NAME,
        "frames_used": idx,
        "frame_vectors_shape": list(fvec.shape),
        "video_vector_shape": list(vvec.shape),
        "video_vector_first10": [round(v, 4) for v in vvec[:10].tolist()],
        "text_match": {t: {"cosine": round(c, 4), "prob": round(p, 4)}
                       for t, c, p in zip(TEXTS, cos.tolist(), probs.tolist())},
        f"per_frame_cosine_to '{TEXTS[0]}'": [round(v, 4) for v in per_frame.tolist()],
        "seconds": round(time.time() - t0, 1),
    }
    best = max(clip_res["text_match"], key=lambda t: clip_res["text_match"][t]["prob"])
    print(f"      video vector {tuple(vvec.shape)}; best text: '{best}' "
          f"({clip_res['text_match'][best]['prob']:.1%})")

    # Stage 3
    mae_res = None
    if not args.skip_videomae:
        print("[3/4] VideoMAE action recognition")
        t0 = time.time()
        model, proc = videomae_classify.load()
        imgs, idx = video.sample(frames, 16)
        top, pooled = videomae_classify.classify(model, proc, imgs)
        np.save(out_dir / "videomae_vector.npy", pooled.numpy())  # (768,)
        mae_res = {
            "model": videomae_classify.MODEL_NAME,
            "frames_used": idx,
            "top5": [{"label": l, "prob": round(p, 4)} for l, p in top],
            "vector_shape": list(pooled.shape),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"      top label: '{top[0][0]}' ({top[0][1]:.1%})")
    else:
        print("[3/4] VideoMAE skipped")

    # Stage 4
    print("[4/4] saving outputs/")
    result = {
        "clip": {"source": str(args.video.name), "start_s": args.start, "seconds": args.seconds,
                 "frames": len(frames), "fps": round(fps, 2)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu"},
        "clip_model": clip_res,
        "videomae": mae_res,
    }
    (out_dir / "output.json").write_text(json.dumps(result, indent=2))
    print("      done -> outputs/output.json, *.npy")


if __name__ == "__main__":
    main()

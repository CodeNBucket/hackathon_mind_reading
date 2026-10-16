"""Stage 1: video file -> 3-second clip -> list of frames (PIL images)."""
from pathlib import Path

import av
import numpy as np


def cut_clip(source: Path, target: Path, start: float, seconds: float) -> None:
    """Copy [start, start+seconds) of `source` into `target` (.mp4). No ffmpeg needed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as src, av.open(str(target), "w") as dst:
        vin = src.streams.video[0]
        vout = dst.add_stream("mpeg4", rate=vin.average_rate)
        vout.width, vout.height, vout.pix_fmt = vin.width, vin.height, "yuv420p"
        vout.bit_rate = 2_000_000
        for frame in src.decode(vin):
            if frame.time is None or frame.time < start:
                continue
            if frame.time >= start + seconds:
                break
            out = frame.reformat(format="yuv420p")
            out.pts = None  # let the encoder restart timestamps at 0
            for pkt in vout.encode(out):
                dst.mux(pkt)
        for pkt in vout.encode():
            dst.mux(pkt)


def read_frames(path: Path):
    """Return (frames, fps). A 3 s clip at 30 fps gives 90 frames."""
    with av.open(str(path)) as c:
        fps = float(c.streams.video[0].average_rate)
        frames = [f.to_image() for f in c.decode(video=0)]
    return frames, fps


def sample(frames, n: int):
    """Pick n evenly spaced frames. Returns (frames, indices)."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int).tolist()
    return [frames[i] for i in idx], idx

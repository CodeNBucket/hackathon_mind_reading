# demo_video_model

Shows what a pretrained vision model outputs for one 3-second video clip.
For the EEG project, the CLIP vector this produces is the target an EEG decoder would learn to predict.

## Run it (Python 3.11)

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

The first run downloads the models (about 1.5 GB) into `models/`. Later runs reuse them.

Options: `--video other.mp4 --start 5 --seconds 3 --skip-videomae`

## Pipeline

| Stage | File | In → Out |
|---|---|---|
| 1. Clip | `pipeline/video.py` | video file → 3 s clip (`data/clip_3s.mp4`) → 90 frames (30 fps × 3 s) |
| 2. CLIP | `pipeline/clip_embed.py` | 8 evenly spaced frames → 8 × 512 numbers → averaged into 1 × 512 "video vector" |
| 3. VideoMAE | `pipeline/videomae_classify.py` | 16 frames → action label (400 Kinetics classes) + 1 × 768 vector |
| 4. Save | `run.py` | `outputs/` |

## Folders

- `data/`: the source video and the 3 s clip cut from it
- `pipeline/`: one file per stage
- `models/`: downloaded model weights (Hugging Face cache). Delete it to free space; it re-downloads on the next run.
- `outputs/`:
  - `output.json`: readable summary
  - `clip_video_vector.npy`: the 512-number video vector
  - `clip_frame_vectors.npy`: 8 × 512, one vector per frame
  - `videomae_vector.npy`: 768 numbers
- `.venv/`: the Python 3.11 environment. Don't edit it, and don't share it; rebuild it from `requirements.txt` instead.

## Clip source

"Dog-agility-06-05-28-luz-stp.ogv", Wikimedia Commons, CC BY-SA 2.5.
https://commons.wikimedia.org/wiki/File:Dog-agility-06-05-28-luz-stp.ogv

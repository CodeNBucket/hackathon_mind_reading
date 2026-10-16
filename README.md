# hackathon_mind_reading

EEG → image/video decoding for the hackathon. Python 3.11.

## Layout

- `experimentation/`: small, self-contained experiments to learn how the pieces work. Each experiment has its own README, `requirements.txt` and `run.py`.
  - `01_video_to_clip_vector/`: turns a 3 s video into a 512-number CLIP vector, the target an EEG model would learn to predict.

The main pipeline will live outside `experimentation/` once we start on the real data.

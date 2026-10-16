# hackathon_mind_reading

Identify which 3-second video a person watched from their EEG. Hackathon project, Python 3.11.

## Layout

- `experimentation/`: small, self-contained experiments to learn how the pieces work. Each experiment has its own README, `requirements.txt` and `run.py`.
  - `01_video_to_clip_vector/`: turns a 3 s video into a 512-number CLIP vector, the target an EEG model would learn to predict.
  - `02_synthetic_eeg_practice/`: fake EEG data shaped like the challenge, plus the full EEG → video vector → pick-a-video pipeline running on it.

The main pipeline will live outside `experimentation/` once we start on the real data.

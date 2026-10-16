# 03: Look at the real data

The organisers' files are not human-readable:
- `.npz` is a zip of named numpy arrays.
- `.npy` is a single numpy array.

This script opens them and turns one recording into a picture and an Excel-readable table.

```
..\02_synthetic_eeg_practice\.venv\Scripts\python.exe peek_data.py                     # sub-01, video 1
..\02_synthetic_eeg_practice\.venv\Scripts\python.exe peek_data.py --sub 5 --video 35  # sub-05, video 0035.mp4
```

It needs only `numpy` and `matplotlib`; the experiment 02 environment has both. Use `--data <folder>` if the challenge data isn't in the default location.

Output in `out/` (not committed, since it's derived from the organisers' data):
- `recording_sub-XX_vid-N.png`: the 17 electrodes over 3 s for one video
- `recording_sub-XX_vid-N.csv`: the same numbers as a table
- `average_sub-XX.png`: the average over all training videos

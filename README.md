# hackathon_mind_reading

Identify which 3-second video a person watched from their EEG. Hackathon project, Python 3.11.

## Layout

- `PROJECT_PLAN.md`: **start here.** The master plan: Person 1 (EEG) and Person 2 (video + evaluation), phases, tasks and checklists.
- `eeg/PLAN.md`: the EEG side's 3 steps from PCA to advanced, all with the same output format.
- `BRIEF.md`: the official hackathon brief and how each part maps to our plan.
- `experimentation/`: small, self-contained experiments to learn how the pieces work. Each experiment has its own README, `requirements.txt` and `run.py`.
  - `01_video_to_clip_vector/`: turns a 3 s video into a 512-number CLIP vector, the target an EEG model would learn to predict.
  - `02_synthetic_eeg_practice/`: fake EEG data shaped like the challenge, plus the full EEG → video vector → pick-a-video pipeline running on it.

## Base pipeline (runs end to end)

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_all.py
```

Set `MINDREADING_DATA` if the organisers' data isn't in the default folder (see `mindreading/config.py`). The first run takes about 15 minutes, mostly the video features; later runs take about 75 s.

| Notebook | Like the organisers' | Does |
|---|---|---|
| `notebooks/03_video_features` | 01, frames + AlexNet | 8 frames per video → CLIP → one 512-number card per video, for all 1,102 videos |
| `notebooks/04_train_and_validate` | 01, train + check | Preprocesses the EEG, chooses settings on held-out *training* videos, scores the sub-01…04 test sets, and runs a shuffled-label check |
| `notebooks/05_predict_submission` | 02 | Trains the final model, saves `weights/`, checks the format on sub-01 and writes `submission/sub-05/06_ranked_candidate_list.npy` |

The code lives in `mindreading/`; the notebooks only call it. `scripts/build_notebooks.py` regenerates the notebooks.

**Model:** the same direction as the organisers' baseline, with CLIP instead of AlexNet. Video card → ridge regression → predicted EEG (17 electrodes × 0–1 s at 100 Hz). Candidates are ranked by the correlation between the real and the predicted EEG. One model per subject. Each EEG file is z-scored with its own statistics, because the test files are scaled differently from the training files.

**Results** on the sub-01…04 test sets (102 candidates):

| | top-1 | top-5 |
|---|---|---|
| chance | 1.0% | 4.9% |
| organisers' baseline (AlexNet) | ~4% | 11.7% |
| **ours: CLIP + ridge encoder** | **5.1%** | **23.3%** |
| shuffled-label control | 1.5% | 5.1% |

Per subject (top-1 / top-5): sub-01 8.8 / 34.3, sub-02 5.9 / 20.6, sub-03 2.9 / 20.6, sub-04 2.9 / 17.6. With 102 recordings, one recording is about 1 point, so these numbers are noisy.

The settings were chosen on held-out training videos only (`results/settings_search.json`). The test answers were used only to report the scores above.

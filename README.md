# hackathon_mind_reading

Identify which 3-second video a person watched from their EEG. Hackathon project, Python 3.11.

## Layout

| Path | What it is |
|---|---|
| `BRIEF.md` | The official task, the data and the rules |
| `run_all.py` | One command: runs our notebooks 03 → 04 → 05 |
| `notebooks/` | `01`, `02`: the organisers' templates (reference). `03`, `04`, `05`: our pipeline |
| `mindreading/` | The code our notebooks call (see "Code" below) |
| `results/` | `dev_test_scores.json`: the scores |
| `past_trials/` | How we chose the settings: `experiment_log.md` and the experiment code. Not needed to run anything |
| `weights/` | The trained model (required deliverable) |
| `submission/` | `sub-05_ranked_candidate_list.npy` and `sub-06_…`: what we hand in |
| `requirements.txt` | Exact library versions (Python 3.11) |
| `cache/`, `.venv/` | On your computer only (git-ignored): saved features and the Python environment |

### Code (`mindreading/`), in the order the data flows

| File | Does |
|---|---|
| `config.py` | Where the data is, subject lists, constants |
| `data.py` | Reads the organisers' files (`trn`, `tst`, `tst_stim`) |
| `video_features.py` | Videos → CLIP features |
| `features.py` | Videos → VideoMAE, low-level time courses and AST audio features |
| `preprocess.py` | EEG: 100 Hz, channel whitening, z-scoring, 0–1 s window |
| `model.py` | The model: video features → predicted EEG, 3 encoders per subject, scoring, validation |
| `ranking.py` | Scores → the 102 × 102 submission format, plus the format checker |

## Pipeline (runs end to end)

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_all.py
```

**Data:** copy the organisers' challenge data (the folders `trn/`, `tst/`, `tst_stim/`, `stimuli_train/`, `stimuli_test/`) into `data/` inside this repo. It is git-ignored, so it is never committed. If the data is somewhere else, set the environment variable `MINDREADING_DATA` to that folder instead, e.g. in PowerShell: `$env:MINDREADING_DATA="D:\eeg-mind-reading-challenge"`. The first run takes about 1 hour, mostly extracting the video and audio features (cached in `cache/`). Later runs take about 2 minutes. A GPU speeds up feature extraction a lot; for that, install the CUDA build of torch, e.g. `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`.

| Notebook | Does |
|---|---|
| `notebooks/01_preprocess_and_train_model` | Organisers' template, step 1 (AlexNet → PCA → ridge → predicted EEG). Reference only; needs `net2brain` and must run from the data folder |
| `notebooks/02_predict_movies` | Organisers' template, step 2 (rank the 102 test videos). Reference only |
| `notebooks/03_video_features` | Ours, replaces their feature step: CLIP, VideoMAE, low-level time courses and audio (AST) for all 1,102 videos |
| `notebooks/04_train_and_validate` | Ours, replaces their training: EEG preprocessing, validation on held-out *training* videos, the sub-01…04 test score, and a shuffled-label check |
| `notebooks/05_predict_submission` | Ours, replaces their notebook 02: trains the final model, saves `weights/`, checks the format on sub-01, writes `submission/sub-05/06_ranked_candidate_list.npy` |

Our notebooks only call the code in `mindreading/`.

### Model

The same direction as the organisers' baseline (video → predicted EEG → compare with the real EEG), improved step by step:

1. **EEG preprocessing:**
   - 100 Hz.
   - **Channel whitening**: decorrelate the 17 electrodes with each file's own covariance, shrunk 30% towards the identity.
   - z-score per file.
   - 0–1 s window.
2. **Stimulus features** (each block standardised on the training videos, then weighted and concatenated):

   | Block | Weight | Captures |
   |---|---|---|
   | CLIP ViT-B/32, 8 frames | 1.0 | Objects and scenes |
   | VideoMAE layers 4/8/12 | 0.6 | Motion and actions |
   | Low-level time courses every 0.1 s: brightness, contrast, saturation, motion, edges, scene cuts, loudness, sound onsets | 0.6 | Timing the early visual and auditory brain reacts to |
   | AST layer 4 | 0.5 | What the sound is |
3. **Encoders:**
   - Ridge regression from features to the 17 × 100 EEG.
   - A separate regularisation strength for every electrode × time point.
   - Three per subject: own data (weight 1), all 6 subjects (0.25), and the grand-average EEG over subjects (0.25).
4. **Rhythm power (0–3 s):** a fourth encoder predicts how strong each brain rhythm (delta, theta, alpha, beta, low gamma) is per electrode in each 0.5 s bin. After about 1 s the exact wave shape is too variable to predict, but this slower brain state is not, so it uses the later 2 seconds of the recording. It is added to the wave score with weight 0.25.
5. **Ranking:** correlation between the real EEG and each candidate's predicted EEG (wave and rhythm power).

Pretrained models (public, pinned to exact Hugging Face commits in `mindreading/features.py`):
- `openai/clip-vit-base-patch32`
- `MCG-NJU/videomae-base-finetuned-kinetics`
- `MIT/ast-finetuned-audioset-10-10-0.4593`

Trained weights: `weights/ensemble_encoders.joblib`.

### Results on the sub-01…04 test sets (102 candidates)

| | top-1 | top-5 | median rank |
|---|---|---|---|
| chance | 1.0% | 4.9% | 51.5 |
| organisers' baseline (AlexNet) | ~4% | 11.7% | |
| first version: CLIP + ridge | 5.1% | 23.3% | 18.3 |
| wave model only (0–1 s) | 16.7% | 46.1% | 6.3 |
| **final model (+ rhythm power 0–3 s)** | **17.6%** | **45.6%** | **6.6** |
| shuffled-label control | 0.7% | 3.2% | |

Per subject, final model (top-1 / top-5): sub-01 18.6 / 47.1, sub-02 19.6 / 42.2, sub-03 11.8 / 46.1, sub-04 20.6 / 47.1. On held-out training videos, rhythm power adds +3.5 points top-5 (20.9% → 24.4%). On the test sets it is neutral (within noise); see `past_trials/experiment_log.md`, round 9.

All settings were chosen on held-out *training* videos (3 random splits averaged). The sub-01…04 test answers were only used to report these scores, never to choose anything and never for sub-05/06.

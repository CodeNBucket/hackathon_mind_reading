# hackathon_mind_reading

Identify which 3-second video a person watched from their EEG. Hackathon project, Python 3.11.

**Final model on the sub-01…04 test sets (102 candidates): 19.9% top-1, 53.7% top-5** (chance 1% / 5%; organisers' baseline ~4% / 11.7%). A fresh clone run end to end with `run_all.py` reproduces the submitted files byte for byte.

We use only the organisers' data and public pretrained models. Video features predict the EEG each video should evoke (an *encoding model*), and each recording is matched to the candidate whose predicted EEG fits best.

## Layout

| Path | What it is |
|---|---|
| `BRIEF.md` | The official task, the data and the rules |
| `run_all.py` | One command: runs our notebooks 03 → 04 → 05 |
| `notebooks/` | `01`, `02`: the organisers' templates (reference). `03`, `04`, `05`: our pipeline |
| `mindreading/` | The code our notebooks call (see "Code" below) |
| `results/` | `dev_test_scores.json`: the scores; `colnorm.json`: the column-normalisation experiment |
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
| `encoders.py` | The contrastive neural-network encoder (trained for ranking) |
| `ranking.py` | Scores → the 102 × 102 submission format, plus the format checker |

## Pipeline (runs end to end)

Requires **Python 3.11** (`run_all.py` stops on any other version). Windows:

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_all.py
```

Linux / macOS: `python3.11 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, `.venv/bin/python run_all.py`.

On Windows, clone into a short path (e.g. `C:\hackathon_mind_reading`): the `jupyter` package installs deeply nested files, and a long folder path can exceed Windows' 260-character limit during `pip install`.

**Data:** copy the organisers' challenge data (the folders `trn/`, `tst/`, `tst_stim/`, `stimuli_train/`, `stimuli_test/`) into `data/` inside this repo. It is git-ignored, so it is never committed. If the data is somewhere else, set the environment variable `MINDREADING_DATA` to that folder instead, e.g. in PowerShell: `$env:MINDREADING_DATA="D:\eeg-mind-reading-challenge"`. The first run takes about 1 hour 20 minutes (measured on a fresh clone, laptop with an RTX 4070 GPU): about 70 minutes to extract the video and audio features (cached in `cache/`; mostly video decoding on the CPU), about 10 minutes to train, validate and write the submission. Later runs reuse the cache and take about 10 minutes (much slower without a GPU, because of the contrastive networks). A GPU speeds up feature extraction and network training a lot; for that, install the CUDA build of torch, e.g. `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`.

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
   | VideoMAE layers 4/8/12 | 1.0 | Motion and actions |
   | Low-level time courses every 0.1 s over 0–1 s (same as the EEG window): brightness, contrast, saturation, motion, edges, scene cuts, loudness, sound onsets | 0.6 | Timing the early visual and auditory brain reacts to |
   | AST layer 4 | 0.5 | What the sound is |
3. **Encoders.** There are two kinds, each fitted three times per subject: on that subject's own data (weight 1), on all 6 subjects (0.25), and on the grand-average EEG over subjects (0.25).
   - **Ridge regression** from features to the 17 × 100 EEG, with a separate regularisation strength for every electrode × time point.
   - **Contrastive neural network** (`mindreading/encoders.py`). It starts from the ridge solution and is trained for ranking: its predicted EEG must correlate better with the right recording than with the other recordings in the batch (InfoNCE, the loss CLIP is trained with). 3 networks with different seeds are averaged.
   - Final score = ridge part + 2 × contrastive part.
4. **Rhythm power (0–3 s):** a ridge encoder predicts how strong each brain rhythm (delta 1–4, theta 4–8, alpha 8–13, beta 13–30, low gamma 30–45 Hz) is per electrode in each 0.5 s bin of the whole 3 s recording (17 × 5 × 6 = 510 values). After about 1 s the exact wave shape varies too much to predict, but this slower brain state does not, so it brings in the 1–3 s that the wave model ignores. It is added to the wave score with weight 0.25.
5. **Ranking:** correlation between the real EEG and each candidate's predicted EEG (wave and rhythm power).
6. **Column normalisation (hubness fix):** some videos score high with almost every recording and crowd the top ranks. Each candidate's score is z-scored with that candidate's mean and spread over the subject's *training* recordings ("is this score unusual for this video?"). It uses training data only, and each test recording is ranked on its own.

Pretrained models (public, pinned to exact Hugging Face commits in `mindreading/features.py`):
- `openai/clip-vit-base-patch32`
- `MCG-NJU/videomae-base-finetuned-kinetics`
- `MIT/ast-finetuned-audioset-10-10-0.4593`

Trained weights: `weights/shared.joblib` (encoders shared by all subjects), `weights/sub-05.joblib`, `weights/sub-06.joblib` (float16; about 50, 26 and 26 MB, each under GitHub's 100 MB limit).

### Results on the sub-01…04 test sets (102 candidates)

| | top-1 | top-5 | median rank |
|---|---|---|---|
| chance | 1.0% | 4.9% | 51.5 |
| organisers' baseline (AlexNet) | ~4% | 11.7% | |
| first version: CLIP + ridge | 5.1% | 23.3% | 18.3 |
| wave model only (0–1 s) | 16.7% | 46.1% | 6.3 |
| + rhythm power 0–3 s | 18.6% | 46.1% | 6.5 |
| + contrastive encoder, VideoMAE weight 1.0 | 19.6% | 51.7% | 5.5 |
| **final model (+ column normalisation)** | **19.9%** | **53.7%** | **5.0** |
| shuffled-label control | 1.0% | 4.9% | |

Per subject, final model (top-1 / top-5): sub-01 25.5 / 60.8, sub-02 14.7 / 46.1, sub-03 11.8 / 52.0, sub-04 27.5 / 55.9.

On held-out training videos, rhythm power raised top-5 from 20.9% to 24.7%, and the contrastive encoder from 24.7% to 28.3%. On 5 fresh splits never used for any choice (the unbiased estimate): wave model only 6.1% / 21.2%, + rhythm power 7.4% / 23.4%, + contrastive encoder 8.1% / 26.3%, final model with column normalisation **8.4% top-1 / 27.0% top-5**. Every step is in `past_trials/experiment_log.md`.

All settings were chosen on held-out *training* videos (3 random splits averaged). The sub-01…04 test answers were only used to report these scores, never to choose anything and never for sub-05/06.

### What we tried that did not help

All of these were tested on held-out training videos; details are in `past_trials/experiment_log.md`.
- **EEG filtering:** band-pass and low-pass filters. Common average reference helped at first, but channel whitening replaced it.
- **Longer EEG windows (0–1.5 to 0–3 s) and a time-resolved model (TRF) on 1–3 s:** both made the ranking worse. The later EEG helps only through rhythm power.
- **A bigger image model (CLIP ViT-L/14, 16 frames):** no gain. Motion, low-level timing and sound add more than a stronger object recogniser.
- **Other PCA sizes and single VideoMAE layers;** block weights other than the final ones. A VideoMAE weight above 1.0 looked better on the tuning splits but not on fresh ones.
- **Other encoders:** kernel ridge (top-5 22.7% vs 24.7% for ridge). A neural network trained to reproduce the EEG did worse than ridge (17.6% vs 22.6%) and hurt the ensemble. Only the network trained for ranking (contrastive loss) helped.

### How we guarded against overfitting

- **Choosing settings:** every setting was chosen on held-out *training* videos (15% hidden, 3 random splits). The sub-01…04 test answers were used only to report scores.
- **Fresh splits:** the final choices were re-checked on 5 new random splits that no choice had used. This revealed that a VideoMAE weight of 1.5 was split luck, so 1.0 is used.
- **Shuffled labels:** training with shuffled video labels gives chance-level scores (1.0% / 4.9%), so the scores come from the real link between videos and EEG.
- **Early stopping:** the neural networks stop on held-out training videos, and 3 seeds are averaged.

### References

- **Encoding models with ridge regression:** Naselaris, Kay, Nishimoto & Gallant (2011), *Encoding and decoding in fMRI*, NeuroImage. Hoerl & Kennard (1970), *Ridge regression*, Technometrics.
- **Temporal response functions for EEG:** Crosse et al. (2016), *The multivariate temporal response function (mTRF) toolbox*, Frontiers in Human Neuroscience.
- **Channel whitening (multivariate noise normalisation):** Guggenmos, Sterzer & Cichy (2018), NeuroImage. Shrinkage covariance: Ledoit & Wolf (2004), Journal of Multivariate Analysis.
- **Video and audio features:**
  - Radford et al. (2021), *CLIP*, ICML.
  - Tong et al. (2022), *VideoMAE*, NeurIPS.
  - Gong, Chung & Glass (2021), *AST: Audio Spectrogram Transformer*, Interspeech.
- **Contrastive training for brain decoding:**
  - van den Oord, Li & Vinyals (2018), *Representation learning with contrastive predictive coding* (InfoNCE).
  - Défossez et al. (2023), *Decoding speech perception from non-invasive brain recordings*, Nature Machine Intelligence.
  - Song et al. (2024), *Decoding natural images from EEG for object recognition*, ICLR.
- **Hubness and column normalisation:** Radovanović, Nanopoulos & Ivanović (2010), *Hubs in space*, JMLR. Dinu, Lazaridou & Baroni (2015), *Improving zero-shot learning by mitigating the hubness problem*, ICLR workshop.
- **Seed ensembles:** Lakshminarayanan, Pritzel & Blundell (2017), *Deep ensembles*, NeurIPS.

# Project plan: EEG → which video?

This is the master plan to follow. The other docs go deeper on one part each:
- `BRIEF.md`: the official task, the real data shapes, the rules and the exact submission format
- `eeg/PLAN.md`: Person 1's three models in detail

Work days are **17–18 October 2026**, so the phase times below are rough guides in hours of working time. The real data is 17 channels × 1500 samples (500 Hz) per recording, 1,000 training videos with sound, and **102 new test videos**; see `BRIEF.md`.

| | Person 1 | Person 2 |
|---|---|---|
| Role | **EEG**: brain signal → predicted video card | **Video + evaluation**: videos → cards, scoring, demo |
| Main folder | `eeg/` | `visual/` |
| Owns the insight on | *when* and *where* in the brain (timing, electrodes) | *which model* is most brain-like (RSA, surprise) |

---

## 1. The task in one picture

```
                 Person 1                                   Person 2
EEG trial ─► clean ─► features ─► translator ─► predicted card (512)
                                                      │
                                                      ▼  compare (cosine similarity)
video 1..N ─────────────────────► CLIP / VideoMAE / V-JEPA 2 ─► video cards (N × 512)
                                                      │
                                                      ▼
                                   closest card = predicted video → top-1 / top-5 score
```

- **Card:** a list of numbers (512 for CLIP) describing a video. The single numbers have no names; only the distance between cards matters. Similar videos get similar cards.
- **Translator:** a model learned from trials where we know which video was shown. It converts EEG into a card.
- **Score:** how often the right video is the closest (top-1) or among the 5 closest (top-5). With the real 102 candidates, chance is 1.0% / 4.9%.

## 2. The contract (change only together, by pull request)

| Item | Value |
|---|---|
| Card model | CLIP `openai/clip-vit-base-patch32`, 8 evenly spaced frames per 3 s clip, averaged, normalised to length 1 → **512 numbers** |
| Video cards file | `shared/video_vectors/<model>.npy` (videos × 512) + `shared/video_vectors/ids.csv` |
| Prediction file | `shared/predictions/<method>_<heldout>.npy` (trials × 512, normalised) + `.csv` with `subject, trial, true_video_id` |
| Validation | hold out ~15% of the *training videos* for every subject for model and hyperparameter selection. Report the final score on the `sub-01`…`sub-04` test sets (allowed for validation only, never as templates for `sub-05`/`sub-06`). Also leave-one-subject-out on the training data as a robustness check. |
| Score | top-1, top-5, per held-out subject and mean |
| Submission | `sub-05_ranked_candidate_list.npy`, `sub-06_ranked_candidate_list.npy`: int arrays 102 × 102 of video ids 1001–1102, rows in the original `tst` order. Built and checked by `shared/ranking.py`. |
| Python | 3.11, pinned `requirements.txt` per folder |

If the card model changes (e.g. VideoMAE with 768 numbers), Person 2 publishes the new cards file and Person 1 retrains. Nothing else changes.

When the organisers' template scripts arrive, their data loading and submission format win. We adapt the contract to them in phase 1.

## 3. Repo layout (target)

```
BRIEF.md  PROJECT_PLAN.md  README.md
eeg/                Person 1
  PLAN.md
  data.py           load + clean EEG (uses organiser template)
  features.py       electrodes, time bins, PCA
  step1_ridge.py    step2_mlp.py    step3_eegnet.py
  predict.py        EEG trials in → predicted cards out (final model)
  weights/          saved weights (required deliverable)
  results/          decoding-over-time curve, head maps
visual/             Person 2
  embed_videos.py   videos → cards (CLIP, VideoMAE, V-JEPA 2, simple features)
  evaluate.py       predictions file → top-1/top-5, confusion table, top-5 per trial
  rsa.py            brain vs model similarity analyses
  demo.py           predictions → videos shown side by side
  results/
shared/
  video_vectors/    cards (Person 2 writes)
  predictions/      predicted cards (Person 1 writes)
run_all.py          end to end: data → models → scores → figures
experimentation/    03_look_at_real_data: viewer for the organiser files
data_raw/           organiser data, NOT in git
```

---

## 4. Phases and tasks

Tick the boxes as you go. "Done when" is the check that a task is finished.

### Phase 0: before the data arrives (now)

**Person 1 (EEG)**
- [ ] Write `eeg/step1_ridge.py` against the fake data, producing the prediction-file format above.
  *Done when:* `visual/evaluate.py` scores it.
- [ ] Draft `eeg/step2_mlp.py` (PCA + small MLP + contrastive loss) on fake data.
- [ ] Read 1–2 papers. Search terms: "EEG to CLIP contrastive", "THINGS-EEG zero-shot".

**Person 2 (video + evaluation)**
- [ ] Write `visual/embed_videos.py`: a folder of videos → `shared/video_vectors/clip_b32.npy` + `ids.csv`.
  *Done when:* it runs on 3–5 test clips.
- [ ] Write `visual/evaluate.py` (top-1/top-5, confusion table, top-5 per trial).
  *Done when:* on fake data, the "perfect brain" test (true cards as predictions) gives 100%.
- [ ] Prepare the simple video features: brightness, motion energy (OpenCV), scene cuts (PySceneDetect), face present.
- [ ] Test V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`) on one clip with a GPU (CUDA build of torch). Note memory use and seconds per clip, so phase 2 has no surprises.

**Together**
- [ ] Agree the contract table above. Both read `BRIEF.md`.

### Phase 1: data arrives (≈ hours 0–8)

**Together, first:** read the organisers' template and baseline scripts end to end, run their baseline as is, and write its score in the results table. That's the first number to beat.

**Person 1**
- [ ] Load the organiser data with their template. Write down the shapes: subjects, trials, electrodes, time points, sampling rate, how the trial → video table looks.
- [ ] Cleaning in `eeg/data.py`: baseline correction, 0.1–40 Hz filter, blink handling, per-subject scaling.
- [ ] Run step 1 (PCA + ridge) on the held-out training videos, then on the `sub-01`…`sub-04` test sets.
  *Done when:* the first real top-1/top-5 is in the results table of `eeg/PLAN.md`.

**Person 2**
- [ ] Run `embed_videos.py` on all challenge videos. Commit `shared/video_vectors/` **first thing**, because Person 1 needs it for step 1.
- [ ] Extract features for all 1,000 training and 102 test videos, **both visual and audio** (the videos have sound). The test videos are new, so only the vector route works.
- [ ] Score Person 1's first predictions with `evaluate.py`.

**Together, end of phase 1:** both checks pass. The perfect brain gives 100%, and training with shuffled labels drops to chance. Then we have a first honest score.

### Phase 2: better models (≈ hours 8–24)

**Person 1**
- [ ] Step 2: PCA + MLP + contrastive loss.
- [ ] Step 3: EEGNet-style network + contrastive loss (braindecode / PyTorch). Use augmentation and early stopping.
- [ ] Psychology-based feature variants (from what's known about vision): visual electrodes only, time windows around P1/N1, N170, P300, alpha power.
- [ ] Decide between one model for all 6 subjects (with per-subject scaling) and per-subject models. `sub-05`/`sub-06` have ~900 labeled training recordings each, so both are possible.
- [ ] Fill the results table for every step and variant.
  *Done when:* each row has scores for all 4 held-out subjects plus the shuffled control.

**Person 2**
- [ ] More card models: VideoMAE (768), bigger CLIP (ViT-L/14), CLIP per second (a card per 1 s instead of per clip).
- [ ] World model: V-JEPA 2 cards plus a "surprise over time" curve per video (GPU, CUDA torch).
- [ ] Report which card model gives Person 1 the best score with the step 1 model. Run it as a quick sweep, then propose a contract change if one clearly wins.
- [ ] `demo.py` v1: for a trial, play the predicted video next to the true one.

**Together, mid-phase:** pick the card model and the best EEG step. Freeze the contract.

### Phase 3: insights, the judged part (≈ hours 24–40)

**Person 1**
- [ ] Decoding over time: accuracy per 50 ms window. Shows *when* the brain "knows".
- [ ] Head map: which electrodes carry the information (MNE topomap).
- [ ] Repetition effect: response on the 1st vs later viewings of the same video.

**Person 2**
- [ ] RSA, which model is most brain-like: correlate the brain's video × video similarity table (from Person 1's EEG per time window) with each model's table (simple features, CLIP, VideoMAE, V-JEPA 2).
- [ ] Time vs depth: same, but per model layer. Does early EEG match early layers?
- [ ] Surprise: V-JEPA 2 surprise curve vs the EEG response over time.
- [ ] Same confusions? Compare the EEG decoder's confusion table with the models' similarity.

**Together:** pick the 3–4 strongest findings and write one slide each: question → figure → one-sentence answer.

### Phase 4: submission (last ≈ 8 hours)

**Person 1**
- [ ] Final model in `eeg/predict.py`, weights in `eeg/weights/`, retrained on the training recordings of all 6 subjects.
- [ ] Apply to the `sub-05`/`sub-06` test recordings. Write the two 102 × 102 files with `shared/ranking.py`, and first run the same code on `sub-01` to check the format against `tst_stim`.

**Person 2**
- [ ] `run_all.py`: one command from raw data to scores and figures.
- [ ] Fresh-environment test: new Python 3.11 venv, `pip install -r requirements.txt`, `run_all.py` runs with no manual steps.
- [ ] `demo.py` final + slides.

**Together**
- [ ] README: how to reproduce, results table, the insight figures.
- [ ] Rehearse the 5-minute story: task → our pipeline → score → what brain vs AI taught us.

---

## 5. How we work together

- **Branches:** `p1/<topic>` and `p2/<topic>`. Small pull requests into `main`, every few hours. The other person skims each one.
- **Folders:** Person 1 commits in `eeg/`, Person 2 in `visual/`. `shared/` and the contract change only through a pull request both have seen.
- **Data:** organiser data lives in `data_raw/` (git-ignored). Never commit raw EEG or videos from the challenge.
- **10-minute sync every ~4 hours:** current score, next step, anything blocking. Update the results table in `eeg/PLAN.md`.
- **Decisions:** write one line in the README under "Decisions" (date, what, why) so the reasoning is reproducible.

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Template differs from our plan | Their format wins. Adapt `eeg/data.py` and the contract in phase 1. |
| Advanced models don't beat ridge | Submit ridge (or an ensemble) and present the comparison as a finding. |
| Low accuracy on `sub-05`/`sub-06` | Per-subject scaling, train on their own training recordings, fewer and more robust features. |
| V-JEPA 2 too heavy or slow | Skip the world-model cards; keep only the surprise curve on a subset, or drop it. |
| Not enough time for all insights | Priority order: decoding over time → head map → which model is most brain-like → the rest. |

## 7. Definition of done (submission checklist)

- [ ] `sub-05_ranked_candidate_list.npy` and `sub-06_ranked_candidate_list.npy` (102 × 102 ints, ids 1001–1102, unsorted row order)
- [ ] Report, slides and a 5-minute presentation (approach, diagrams, results, failure cases, lessons)
- [ ] Training pipeline + weights in the repo, reproducible in a fresh Python 3.11 venv
- [ ] Results table: every step × `sub-01`…`sub-04`, with the shuffled control
- [ ] 3–4 insight figures with a one-line answer each
- [ ] Demo: an EEG trial → the predicted video next to the true one

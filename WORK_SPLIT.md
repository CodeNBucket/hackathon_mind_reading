# Work split: EEG person and video person

Task: from a person's EEG, identify which 3-second video they watched, and show it.

```
EEG trial ──► [EEG person: EEG model] ──► predicted video vector ──► [Video person: search / generator] ──► video
                                                   ▲
                                     the only thing both sides share
```

## The contract (agreed once, changed only together)

| Item | Value |
|---|---|
| Video vector | CLIP `openai/clip-vit-base-patch32`. Take 8 evenly spaced frames per 3 s clip, average them, normalise to length 1. Gives 512 numbers (experiment 01). |
| Exchange format | `.npy` of shape `(N, 512)` plus a `.csv` with the N video ids in the same order |
| Candidates | all videos in the challenge's test set |
| Score | top-1 and top-5: is the right video among the 1 or 5 closest candidates? With 20 videos, chance is 5% / 25%. |

When the organisers' template arrives, its layout and submission format win, and we adapt this plan to it.

Exchange files go in `shared/`. Code goes in `eeg/` or `visual/`. Raw data goes in `data_raw/` (not in git).

Until the organisers' data arrives, both sides practise on `experimentation/02_synthetic_eeg_practice` (fake EEG, same shape as the challenge). Its `video_embeddings_toy.npy` gets replaced by real vectors from the video person.

---

## EEG person

**Job:** turn a brain signal into a predicted video vector.

**Goal:** predictions that land closest to the video the person actually watched, on subjects the model has never seen. Success is measured by the shared score.

**Steps**
1. Load the EEG trials: 33 electrodes × 320 time points per trial in the mock.
2. Clean the data:
   - remove or repair blink trials
   - baseline-correct
   - average repetitions of the same video where allowed
3. Features: back-of-head electrodes and 50 ms time bins, then PCA. Experiment 02 already does this and gives 650 → 40 numbers.
4. Baseline: ridge regression from the EEG features to the 512-number video vectors in `shared/video_vectors/`.
5. Improve:
   - better time windows
   - small neural net
   - contrastive (CLIP-style) loss
   - leave-one-subject-out validation
6. Each time the score improves, export the predictions.

**Expected output**
- `eeg/predict.py`: EEG trials in, `(trials, 512)` predicted vectors out.
- `shared/predictions/<method>.npy` + a `.csv` listing, for each trial, its subject and true video id.
- The final model's saved weights in `eeg/weights/`. Sharing weights is a required deliverable.
- For the insight story, a decoding-over-time curve: how much each 50 ms window and each brain area tells us about the video.
- A line in the README with the method and its score. Experiment 02's ridge pipeline on fake data gives 34% top-1, 85% top-5 out of 20. That is the bar to beat once real data arrives.

---

## Video person

**Job:** turn videos into vectors, turn predicted vectors back into videos, and measure how good it is.

**Goal:** a pipeline that, given any predictions file, scores it and shows the decoded videos. It must be ready before the EEG model is good, so the EEG person sees results immediately.

**Steps**
1. **Day 1 (blocks the EEG person, do it first):**
   - Run experiment 01's `pipeline/clip_embed.py` over every challenge video, or the mock video list until the real ones arrive.
   - Save `shared/video_vectors/vectors.npy` (videos, 512) and `ids.csv`.
2. `visual/evaluate.py`:
   - input: a predictions file
   - output: top-1 / top-5, a confusion table (which videos get mixed up), and the top-5 candidates per trial
3. Self-test: feed the real video vectors as predictions (a "perfect brain"). It must score 100%.
4. Try other video vectors and report which ones the EEG person should target:
   - CLIP with more frames
   - VideoMAE (768 numbers)
   - per-second vectors instead of one per clip

   Any change of vector goes through the contract.

   Also try two non-CLIP baselines:
   - simple features: brightness, motion energy, scene cuts. They're cheap, and the brain reacts strongly to them, so they keep us honest.
   - a world-model surprise curve: how badly V-JEPA 2 predicts each next moment of the video. We then check whether the EEG reacts at the same moments.
4b. **Brain vs AI analysis (this is the judged "insight").** RSA: for each EEG time window, compare which videos the brain treats as similar with which videos each model and layer treats as similar. Plot model/layer against time.
5. Demo, from simplest to most impressive:
   - a. **Retrieval:** play the closest candidate video next to the true one.
   - b. **Generation (stretch):** generate a new image or video from the predicted vector (needs a GPU and possibly a bigger CLIP).

**Expected output**
- `shared/video_vectors/*` (day 1)
- `visual/evaluate.py` + score report
- `visual/demo.py`: predictions in → videos shown side by side
- `visual/results/`: the video-vector comparison table, the model/layer × time RSA figure and the surprise-curve comparison

---

## Connecting the two

There is no big merge at the end. The two sides connect through files from day 1.

| When | EEG person | Video person | Connection |
|---|---|---|---|
| Day 1 | load and clean EEG (mock first) | `shared/video_vectors` | agree on the contract table above |
| Day 2 | ridge baseline → `shared/predictions/` | `evaluate.py`, perfect-brain test = 100% | **first real score** |
| Middle | better models, new prediction files | better video vectors, demo | every prediction file is scored the same way. A new video vector means the EEG person retrains. |
| End | best model behind `eeg/predict.py` | `visual/demo.py` | `run_demo.py` = `eeg/predict.py` → `visual/demo.py` |

For the slides, the EEG person brings the decoding-over-time curve and the video person brings the brain-vs-AI figures. Together they tell the story.

Final demo flow: pick an EEG trial → EEG model predicts 512 numbers → closest video(s) → play it next to the video the person really watched, plus the overall top-1/top-5 score.

Rules that keep you unblocked:
- Never change the contract alone.
- Each person commits only in their own folder, on their own branch, and opens small pull requests into `main`.
- `shared/` changes go through a pull request the other person sees.

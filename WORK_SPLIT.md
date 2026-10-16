# Work split: EEG person and video person

```
EEG trial ──► [EEG person: EEG model] ──► 512 numbers ──► [Video person: search / generator] ──► image or video
                                              ▲
                                  the only thing both sides share
```

## The contract (agreed once, changed only together)

| Item | Value |
|---|---|
| Embedding model | `openai/clip-vit-base-patch32`, image embedding, normalised to length 1 |
| Vector size | 512 |
| Exchange format | `.npy` of shape `(N, 512)` plus a `.txt` with the N image ids in the same order |
| Test set | the 200 THINGS-EEG test images, in the dataset's own order |
| Score | top-1 and top-5 accuracy: is the right image among the 1 or 5 closest out of 200? Chance is 0.5% / 2.5% |

Exchange files go in `shared/`. Code goes in `eeg/` or `visual/`. Raw data goes in `data_raw/` (not in git).

---

## EEG person

**Job:** turn a brain signal into a CLIP vector.

**Goal:** predicted vectors that land close to the real CLIP vector of the image the person saw. Success is measured by the shared score.

**Steps**
1. Download THINGS-EEG preprocessed data for one participant (sub-01).
   - Train: 16,540 images × 4 repetitions × 17 channels × 100 time points.
   - Test: 200 images × 80 repetitions.
2. Average the repetitions and flatten each trial to 1,700 numbers. Standardise (z-score) the features.
3. Baseline: ridge regression from EEG (1,700) to CLIP (512), trained on `shared/clip_targets/train.npy`.
4. Improve:
   - pick the best time window (roughly 60–500 ms) and channels
   - small neural net
   - contrastive (CLIP-style) loss
   - more participants
5. Each time the score improves, export the predictions.

**Expected output**
- `shared/predictions/sub-01_test.npy`: shape (200, 512), test-image order.
- A line in the README with the method and its score. Rough guide:
  - Ridge baseline: a few % top-1. Anything well above 0.5% means the signal is real.
  - Good: 15%+ top-1, 40%+ top-5.

---

## Video person

**Job:** turn a CLIP vector into something you can see, and measure how good it is.

**Goal:** a working pipeline that, given any (N, 512) file, scores it and shows the decoded images or videos. It must be ready before the EEG model is good, so the EEG person can see results immediately.

**Steps**
1. **Day 1 (blocks the EEG person, do it first):**
   - Compute CLIP vectors for the THINGS-EEG images, reusing `experimentation/01_video_to_clip_vector/pipeline/clip_embed.py`.
   - Save `shared/clip_targets/train.npy` (16540, 512), `shared/clip_targets/test.npy` (200, 512), and the matching id lists.
2. `visual/evaluate.py`:
   - input: a predictions file
   - output: top-1 / top-5 accuracy, and a picture grid of the 5 closest images for each test trial
3. Self-test: run `evaluate.py` on the real test vectors (a "perfect brain"). It must score 100%. This proves the pipeline before any EEG is involved.
4. Video output, from simplest to most impressive:
   - a. **Retrieval:** embed a library of short video clips with experiment 01 and show the closest clip.
   - b. **Generation:** feed the vector to an image generator that accepts CLIP image embeddings, then optionally animate the image with an image-to-video model.
     - Most of these generators expect a bigger CLIP (ViT-H/14, 1024 numbers). If we go this way, both people switch the contract together. It only means re-running step 1 with another model name.
5. Final demo script and slides.

**Expected output**
- `shared/clip_targets/*` (day 1)
- `visual/evaluate.py` + the score report
- `visual/demo.py`: predictions in → images/videos out

---

## Connecting the two

There is no big merge at the end. The two sides connect through files from day 1.

| When | EEG person | Video person | Connection |
|---|---|---|---|
| Day 1 | download EEG, load it | `clip_targets` | agree on the contract table above |
| Day 2 | ridge baseline → `predictions/` | `evaluate.py`, perfect-brain test = 100% | **first real score** |
| Middle | better models, new prediction files | video retrieval / generation | every new prediction file is scored and visualised the same way |
| End | best model + `eeg/predict.py` (EEG trial → 512 numbers) | `visual/demo.py` | `run_demo.py` = `eeg/predict.py` → `visual/demo.py` |

Final demo flow: pick a test EEG trial → EEG model predicts 512 numbers → closest image/clip and/or generated video → show it next to the image the person actually saw, plus the overall top-1/top-5 score.

Rules that keep you unblocked:
- Never change the contract alone.
- Each person commits only in their own folder, on their own branch, and opens small pull requests into `main`.
- `shared/` changes go through a pull request the other person sees.

# EEG side: 3 steps from PCA to advanced

Each step changes **one thing** from the step before, so we always know what made the score go up or down.

| Step | Shrink the EEG | Translator | How it learns |
|---|---|---|---|
| 1. Baseline | PCA | ridge regression (linear) | "get close to the right video vector" |
| 2. Better learning | PCA (same as step 1) | small neural net (MLP) | contrastive: "closer to the right video than to the 19 wrong ones" |
| 3. Advanced | learned by the network itself (no PCA) | small EEG network (EEGNet-style, convolutions over electrodes and time) | contrastive (same as step 2) |

## Same output for all three (the contract)

Every step produces exactly the same kind of file, so `visual/evaluate.py` scores all of them the same way:

- `shared/predictions/<step>_<heldout-subject>.npy`, shape **(number of trials, 512)**, each row normalised to length 1
- `shared/predictions/<step>_<heldout-subject>.csv`, one row per trial: `subject, trial, true_video_id`

The 512 comes from the video vectors (CLIP ViT-B/32, see `WORK_SPLIT.md`). If the video person switches the video model, for example to VideoMAE with 768 numbers, all three steps switch with it. Only the last layer changes.

## Shared by all three

**Cleaning (done once, reused)**
- baseline-correct with the 200 ms before the video starts
- band-pass filter 0.1–40 Hz
- detect blinks on the front electrodes (Fp1, Fp2); drop or repair those trials
- standardise each subject separately, which helps on new subjects

**Validation**
- Leave-one-subject-out: train on 3 subjects, test on the 4th, rotate through all 4.
- Report top-1 / top-5 per held-out subject and on average.

**Sanity checks, every run**
- *Perfect brain:* evaluating the true video vectors must give 100%.
- *Shuffled control:* train with the trial → video labels shuffled; the score must drop to chance (5% / 25% with 20 videos). If it doesn't, something leaks.

## Step 1: PCA + ridge (baseline, about 1 hour)

- Features: back-of-head electrodes × 50 ms time bins, from 0 to 3 s.
- PCA fitted on the training subjects only. Try k = 20, 40, 80, 160.
- Ridge regression to the 512 numbers. Choose the regularisation strength by cross-validation inside the training subjects.
- Starting point: `experimentation/02_synthetic_eeg_practice/end_to_end_example.py`.
- Insight output: plot what the top PCA components look like as head maps and timelines.

## Step 2: PCA + small net + contrastive loss

- Same PCA features as step 1. Translator: 2-layer MLP, then 512 outputs normalised to length 1.
- Loss: for each batch, the EEG must be most similar to its own video vector and less similar to the other videos (InfoNCE, the loss CLIP was trained with).
- Changes only the learning rule, so the difference from step 1 tells us whether contrastive training helps.

## Step 3: EEG network, no PCA (advanced)

- Input: cleaned EEG, electrodes × time.
- Network: EEGNet-style. A temporal convolution acts as a learned filter, a spatial convolution learns which electrodes matter, then pooling, then a projection to 512. Kept small for 4 subjects.
- Same contrastive loss as step 2. Regularisation: dropout, early stopping on a held-out training subject, and data augmentation (small time shifts, noise).
- Insight output: which electrodes and moments the network relies on (saliency maps), compared with step 1's PCA maps.
- Stretch: start from a pre-trained EEG foundation model (e.g. LaBraM) instead of random weights.

## Result table we fill in when the real data arrives

| Step | sub-01 | sub-02 | sub-03 | sub-04 | mean top-1 | mean top-5 | shuffled control |
|---|---|---|---|---|---|---|---|
| 1. PCA + ridge | | | | | | | |
| 2. PCA + MLP contrastive | | | | | | | |
| 3. EEGNet contrastive | | | | | | | |

# Challenge brief: Mind Reading Using Machine Learning and EEG

National AI Hackathon 2026 (SchubKI, Constructor Technology). **16–21 October 2026:** 16 opening, **17–18 work**, 19–20 evaluation, 21 closing.

Task: given one EEG recording and a set of candidate videos, **rank the candidates from most to least likely** to have elicited that recording. The organizers' intro video and their two template notebooks are the reference for data loading and the baseline.

## The data (what the organizers gave us)

| Item | Detail |
| --- | --- |
| Subjects | 6: `sub-01`…`sub-04` are the *development* subjects, `sub-05` and `sub-06` are the *held-out* subjects |
| EEG | **17 channels, 500 Hz, 3 s from video onset = 1500 samples**. No pre-stimulus baseline period. |
| `stimuli_train/` | 1,000 videos `0001.mp4`…`1000.mp4`, **with sound** |
| `stimuli_test/` | 102 videos `1001.mp4`…`1102.mp4`, with sound. **All subjects saw the same 102.** No training video is in the test set. |
| `trn/subXX_trn.npz` | dict: key = training video id **without leading zeros** (`"35"` = `0035.mp4`), value = array `17 × 1500`. **850–972 recordings per subject, one per video** (so no repetitions to average). Provided for all 6 subjects, labels included. |
| `tst/subXX_tst.npy` | array `102 × 17 × 1500`, one recording per test video, **in shuffled order** |
| `tst_stim/subXX_tst_stim.npy` | the true video ids (1001–1102) in the same order as `tst`. **Only for `sub-01`…`sub-04`.** |
| Local copy | `C:\Users\canbe\Desktop\eeg-mind-reading-challenge\eeg-mind-reading-challenge` (never committed) |

## What we must submit

- **Predictions:** `sub-05_ranked_candidate_list.npy` and `sub-06_ranked_candidate_list.npy`, each an **integer array of shape 102 × 102** saved with `numpy.save`.
  - Row *i* = the *i*-th recording of `tst/sub-XX_tst.npy`, **in the original shuffled order. Do not reorder.** (The organizers' notebook sorts rows by label for validation; the submission must not.)
  - Each row contains all 102 **video ids (1001–1102), not column indices 0–101**, from most to least likely.
  - Other names, shapes or contents cannot be scored. Check the format by running the same submission code on `sub-01` and scoring it with `tst_stim/sub-01_tst_stim.npy`.
- **Git repository:** source code, training and inference pipeline, **all model weights** needed to reproduce (for public pretrained models, their exact name and version instead), and the predictions. No videos or extracted frames.
- **Report**, **slides** and a **5-minute presentation** covering approach, diagrams, results, failure cases and lessons learned.

## How we're scored

- **Quantitative, 70%:** top-1 and top-5 accuracy on `sub-05` and `sub-06`. Chance is 1/102 ≈ 1.0% for top-1 and 5/102 ≈ 4.9% for top-5. The organizers' own simple baseline gets about 4% top-1 and 11.7% top-5 on held-out data.
- **Qualitative, 30%:** technical soundness and novelty 15%, engineering quality and reproducibility 10%, presentation including failure-case analysis 5%.

## Rules that shape our design

1. Python 3.11.
2. Public pretrained models and extra datasets are allowed if publicly available to everyone and released before the hackathon started.
3. **No test data of any subject for training or fine-tuning.** The *training* recordings of all six subjects may be used for training, including `sub-05` and `sub-06`.
4. The test recordings and labels of `sub-01`…`sub-04` are for **validation and model selection only**, and must **never** be used as reference responses (templates) to identify the `sub-05`/`sub-06` test recordings.
5. Unsupervised normalization of the test recordings and of the video features (e.g. z-scoring) is allowed.
6. The intro video adds that hyperparameter selection should also not touch the test split, and should use a validation split of the training videos. **The written rules and the video differ on this.** Until the organizers answer, select hyperparameters on a held-out part of the *training* videos and use the `sub-01`…`sub-04` test sets only to report the final validation score.

## What this changes in our plan

| Earlier assumption | Reality |
| --- | --- |
| Test candidates might be the training videos | **They are 102 new videos.** Only the "vector" route works: a video-id classifier cannot. |
| 20 candidate videos, chance 5% / 25% | **102 candidates**, chance 1.0% / 4.9% |
| Several repetitions per video to average | **One recording per video** per subject. Noise reduction must come from pooling subjects and models, not from averaging repeats. |
| Leave-one-subject-out as the validation scheme | We also have the held-out subjects' **own labeled training recordings**, so subject-specific models for `sub-05`/`sub-06` are possible. Validate by holding out *training videos* per subject, then confirm on the `sub-01`…`sub-04` test sets. |
| Calibrate on a little data from the new subjects | Not a thing: we have ~900 labeled training recordings for each of them. |
| Video side = visual only | **Videos have sound.** Audio and audio-visual models are explicitly relevant. |
| 33 channels at 100 Hz | **17 channels at 500 Hz**, 1500 samples |
| Only the "decode" direction (EEG → video vector) | The organizers' baseline goes the other way ("encode": video features → PCA + regression → *predicted EEG*, then rank by distance to the real EEG). We try both and combine. |
| Submission: any ranked file | Exactly the two 102 × 102 integer files above, in unsorted row order, with ids 1001–1102 |

## Still unknown (ask the organizers or check the data)

- Which 17 channels these are (names / positions).
- Whether the training folder has recordings for every training video across subjects or only subsets, and how much the subjects overlap.
- The exact deadline for the submission on the 18th, and **which GitHub repository is the "designated" one** (is it `CodeNBucket/hackathon_mind_reading`?).
- The hyperparameter-selection question in rule 6.

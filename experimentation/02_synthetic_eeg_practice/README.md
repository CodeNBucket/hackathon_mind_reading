# 02: Synthetic EEG practice kit

Fake EEG data shaped like the real challenge, so we can build and test the EEG side before the organizers' data arrives. **Everything here is synthetic.** The numbers say nothing about real brains.

`CHALLENGE_BRIEF_mock.md` is an imagined version of the brief the organizers might send.

## What the fake dataset looks like

- 4 subjects, 20 videos, 5 repetitions each, so 400 trials.
- Each trial is 33 electrodes × 320 time points (100 Hz, from 0.2 s before the video until 3 s after it starts), in microvolts.
- Built-in effects to find: P1/N1 responses at the back of the head, a face response (N170), motion and scene-cut responses, alpha rhythm, pink noise, and blinks in 20% of trials.
- `video_embeddings_toy.npy` (20 videos × 16 numbers) stands in for a video model's output. Replace it with real CLIP or VideoMAE vectors from experiment 01.

## Run it (Python 3.11)

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe make_sample_data.py      # 1. builds eeg_epochs.npy and the csv/json files
.venv\Scripts\python.exe walkthrough.py           # 2. looks at the data, saves 3 figures, simple decoder
.venv\Scripts\python.exe end_to_end_example.py    # 3. the full EEG -> video-vector -> pick-a-video pipeline
```

(On Mac/Linux use `python3.11 -m venv .venv` and `.venv/bin/python`.)

## What each script shows

- `walkthrough.py`: why we average repetitions (single trial vs average), the face response by category, and what a blink looks like. Then a nearest-template decoder tested on an unseen subject (leave-one-subject-out). Single trials get 12.0% and 5-repetition averages get 33.8%, against 5% chance.
- `end_to_end_example.py`: the pipeline we plan to use.
  1. Keep the 13 back-of-head electrodes and cut time into 50 ms bins, giving 650 numbers per trial.
  2. PCA down to 40 numbers, fitted on training subjects only.
  3. Take the video vector of each video.
  4. Ridge regression from the 40 EEG numbers to the video vector.
  5. Cosine similarity to all 20 candidate videos, and the closest one is the answer.

  Result on an unseen subject: 34% top-1 and 85% top-5 averaged over subjects (sub-04: 20% / 65%). Chance is 5% and 25%.

Results above were checked on Python 3.11 with the versions in `requirements.txt`.

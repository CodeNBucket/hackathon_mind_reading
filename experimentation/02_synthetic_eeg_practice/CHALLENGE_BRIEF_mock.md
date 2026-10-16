# Mock challenge brief (practice version)

*This is an imagined brief, written to look like what the organizers might send. The data in this folder is synthetic.*

## Task
Each trial is 3.2 s of EEG (200 ms before a video starts, 3 s while it plays) recorded from one person.
For each test trial you get a list of candidate videos. Return the candidate you think the person was watching,
ranked from most to least likely.

## Data you get
| File | Contents |
| --- | --- |
| `eeg_epochs.npy` | float32 array, shape (trials, 33 channels, 320 time points), microvolts, 100 Hz |
| `trials.csv` | one row per trial: `subject`, `trial`, `video_id`, `repetition`, `session`, `has_blink` |
| `videos.csv` | one row per video: `video_id`, `category`, `mean_luminance`, `motion_energy`, `cut_time_s` |
| `info.json` | sampling rate, epoch start/end, channel names, reference, filters already applied |
| `video_embeddings_toy.npy` | (20 videos, 16) toy stand-in for a video model's embedding; replace with real ones |
| *(real event)* `stimuli/*.mp4` | the 3-second videos themselves |

## Splits
- Training: subjects sub-01 to sub-04, all trials labelled.
- Test (real event): 2 new subjects. Part of their data is labelled for calibration; the rest is scored.

## Scoring
- Top-1 accuracy over candidates (chance = 1 / number of candidates), plus top-5 as a secondary metric.
- Judges also score scientific insight, robustness on new subjects, and reproducibility.

## Deliverables
1. Training and inference code (Python 3.11) that reproduces your submission from the raw files.
2. Model weights.
3. A short report or slides: approach, validation scheme, results with chance level, and what the model
   tells you about how brains and artificial networks process video.

## Practice goals for today
1. Run `python walkthrough.py` and read every printed line and figure.
2. Change the time window, channels or averaging and see how leave-one-subject-out accuracy moves.
3. Try to beat the nearest-template decoder with any method you like, without touching the test subject.

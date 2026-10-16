"""Paths and constants shared by every notebook and script."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Organiser data (never committed). Override with the MINDREADING_DATA environment variable.
DATA_DIR = Path(os.environ.get(
    "MINDREADING_DATA",
    r"C:\Users\canbe\Desktop\eeg-mind-reading-challenge\eeg-mind-reading-challenge"))

CACHE_DIR = REPO / "cache"            # video features, preprocessed EEG (git-ignored)
WEIGHTS_DIR = REPO / "weights"        # trained models (committed: required deliverable)
SUBMISSION_DIR = REPO / "submission"  # the two ranked candidate lists (committed)
RESULTS_DIR = REPO / "results"        # scores and figures (committed)

SUBJECTS = [1, 2, 3, 4, 5, 6]
DEV_SUBJECTS = [1, 2, 3, 4]           # have test labels: validation only
HELDOUT_SUBJECTS = [5, 6]             # graded by the organisers

SFREQ = 500                           # Hz in the raw files
N_CHANNELS = 17
TRAIN_IDS = list(range(1, 1001))      # stimuli_train/0001.mp4 ... 1000.mp4
TEST_IDS = list(range(1001, 1103))    # stimuli_test/1001.mp4 ... 1102.mp4
SEED = 0

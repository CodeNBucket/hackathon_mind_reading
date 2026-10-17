"""Paths and constants shared by every notebook and script."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Organiser data (never committed): put it in data/ inside the repo (folders trn/, tst/,
# tst_stim/, stimuli_train/, stimuli_test/), or point the MINDREADING_DATA environment
# variable at the folder that contains them.
DATA_DIR = Path(os.environ.get("MINDREADING_DATA", REPO / "data"))


def check_data_dir(data_dir=None):
    """Stop with a clear message if the organisers' data is not where we expect it."""
    d = Path(data_dir or DATA_DIR)
    if not (d / "trn").is_dir():
        raise FileNotFoundError(
            f"Organiser data not found in {d}. Copy the challenge data (trn/, tst/, tst_stim/, "
            f"stimuli_train/, stimuli_test/) into {REPO / 'data'}, or set the MINDREADING_DATA "
            f"environment variable to the folder that contains them.")
    return d

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

"""Loading the organisers' files.

trn/sub-XX_trn.npz       dict  video id (str, no leading zeros) -> EEG (17, 1500)
tst/sub-XX_tst.npy       array (102, 17, 1500), rows in shuffled order
tst_stim/sub-XX_tst_stim.npy  true video id per tst row (sub-01..04 only)
"""
from pathlib import Path

import numpy as np

from .config import DATA_DIR


def load_train(sub: int, data_dir: Path = DATA_DIR):
    """Returns (video_ids [n], eeg [n, 17, 1500]) sorted by video id."""
    z = np.load(Path(data_dir) / "trn" / f"sub-{sub:02d}_trn.npz")
    ids = np.array(sorted(z.files, key=int), dtype=int)
    eeg = np.stack([z[str(i)] for i in ids]).astype(np.float32)
    return ids, eeg


def load_test(sub: int, data_dir: Path = DATA_DIR):
    """Returns (eeg [102, 17, 1500] in the original shuffled order, labels [102] or None)."""
    eeg = np.load(Path(data_dir) / "tst" / f"sub-{sub:02d}_tst.npy").astype(np.float32)
    stim = Path(data_dir) / "tst_stim" / f"sub-{sub:02d}_tst_stim.npy"
    labels = np.load(stim).astype(int) if stim.exists() else None
    return eeg, labels


def video_path(video_id: int, data_dir: Path = DATA_DIR) -> Path:
    folder = "stimuli_train" if video_id <= 1000 else "stimuli_test"
    return Path(data_dir) / folder / f"{video_id:04d}.mp4"

"""One command from raw data to submission files.

    .venv\\Scripts\\python.exe run_all.py

Runs notebooks/03 -> 04 -> 05 in order and saves them with their outputs.
Set MINDREADING_DATA if the organisers' data folder is somewhere else.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent / "notebooks"
ORDER = ["03_video_features", "04_train_and_validate", "05_predict_submission"]


def main():
    for name in ORDER:
        path = NB_DIR / f"{name}.ipynb"
        t0 = time.time()
        print(f"running {path.name} ...", flush=True)
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbformat.write(nb, path)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    print("submission files are in submission/")


if __name__ == "__main__":
    sys.exit(main())

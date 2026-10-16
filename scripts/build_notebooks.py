"""Generate notebooks/03-05 from the cell lists below (keeps them in sync with the code).

    .venv\\Scripts\\python.exe scripts\\build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = """import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # repo root, so `mindreading` imports work
import numpy as np
import matplotlib.pyplot as plt
from mindreading.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, SUBMISSION_DIR
print('data folder:', DATA_DIR)"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NB03 = [
    md("""
# 03 · Video features (the "cards")

The organisers' notebook 01 describes each video with **AlexNet** (an image network from 2012).
We use **CLIP** instead: a newer model that was trained to match images with text, so its
numbers capture *what is in the picture* (dog, face, car, crowd ...).

For each of the **1,102 videos** (1,000 training + 102 test):
1. take 8 evenly spaced frames from the 3 s clip,
2. turn each frame into 512 numbers with CLIP,
3. average the 8 → one card of 512 numbers per video.

The result is cached in `cache/clip_b32.npz`, so this only runs once (a few minutes on CPU).
"""),
    code(SETUP),
    code("""
from mindreading.video_features import extract_clip, lookup
feats = extract_clip()
print('video ids   :', feats['ids'][:3], '...', feats['ids'][-3:])
print('per frame   :', feats['frames'].shape, '(videos x frames x numbers)')
print('one per clip:', feats['mean'].shape, '(videos x numbers)')
"""),
    md("""
### Sanity check: do similar videos get similar cards?
For three test videos, list the training videos with the most similar card (cosine similarity, 1 = identical).
Open the video files to see whether the neighbours really look alike.
"""),
    code("""
train_ids, test_ids = np.arange(1, 1001), np.arange(1001, 1103)
A, B = lookup(feats, test_ids), lookup(feats, train_ids)
sim = A @ B.T
for row in [0, 1, 2]:
    best = np.argsort(-sim[row])[:3]
    print(f'test video {test_ids[row]}: most similar training videos',
          [(int(train_ids[b]), round(float(sim[row, b]), 2)) for b in best])
"""),
]

NB04 = [
    md("""
# 04 · Train and validate

Same idea as the organisers' notebook 01, extended. Everything here uses **training data**,
plus the test answers of sub-01…04 **only to report a score** (never for the held-out subjects).

**Preprocessing** (`mindreading/preprocess.py`)
- 500 → 100 measurements per second (like the organisers).
- Normalise each file with its *own* statistics. The test files are scaled differently from
  the training files, so this matters.
- Keep a time window after the video starts.

**Two models** (`mindreading/models.py`), both PCA/ridge, both linear:
- **Encoder** (organisers' direction): video card → *predicted EEG*. Score a recording against a
  candidate video by correlating the real EEG with the EEG predicted for that video.
- **Decoder** (our direction): EEG → *predicted card*. Score by cosine similarity between the
  predicted card and each candidate's card.
- **Combined**: put both scores on the same scale and add them.

**Pooling**: by default one model is trained on the training recordings of **all 6 subjects**
(about 5,650 recordings). The rules allow this, including sub-05/06's *training* files.
"""),
    code(SETUP),
    code("""
from mindreading.data import load_train
from mindreading.preprocess import downsample, normalise
ids, raw = load_train(1)
x = normalise(downsample(raw))
t = np.arange(x.shape[-1]) / 100
plt.figure(figsize=(10, 3))
plt.plot(t, x.mean(0).T, lw=0.8)
plt.title('sub-01: average over all training videos, one line per electrode')
plt.xlabel('seconds after video start'); plt.show()
"""),
    md("""
### Choose the settings (on held-out training videos only)
`scripts/search_settings.py` hides 15% of the training videos, trains on the rest, and checks
how well the hidden ones are ranked. It tries different time windows, normalisations, pooled vs
per-subject models, and averaged vs per-frame cards. We load its results here, or run it if
they're missing.
"""),
    code("""
import subprocess
path = RESULTS_DIR / 'settings_search.json'
if not path.exists():
    subprocess.run([sys.executable, str(Path.cwd().parent / 'scripts' / 'search_settings.py')], check=True)
search = json.loads(path.read_text())
print(f"{'setting':28s} {'encoder top5':>12s} {'decoder top5':>12s} {'combined top5':>13s}")
for name, r in search.items():
    s = r['scores']
    print(f"{name:28s} {s['encoder']['top5']:12.3f} {s['decoder']['top5']:12.3f} {s['combined']['top5']:13.3f}")
pairs = [(r['scores'][m]['top5'], name, m) for name, r in search.items() for m in r['scores']]
_, best_name, best_method = max(pairs)
print(f'\\nchosen: {best_name!r} with the {best_method}')
"""),
    code("""
from mindreading.pipeline import Config, score_dev_test, summarise
from mindreading.video_features import extract_clip
chosen = dict(search[best_name]['config'])
if best_method != 'combined':
    chosen['methods'] = [best_method]          # train only the model that won
cfg = Config(**chosen)
(RESULTS_DIR / 'chosen_config.json').write_text(json.dumps(chosen, indent=2))
feats = extract_clip()
cfg
"""),
    md("""
### Score on the sub-01…04 test sets (102 candidates, chance = 1.0% top-1, 4.9% top-5)
The organisers' AlexNet baseline: about **4% top-1, 11.7% top-5**.
"""),
    code("""
res = score_dev_test(cfg, feats)
for s, r in res.items():
    print(f'sub-0{s}: ' + '   '.join(f"{m} top1 {v['top1']:.3f} top5 {v['top5']:.3f}" for m, v in r.items()))
mean = summarise(res)
print('\\nmean:   ' + '   '.join(f"{m} top1 {v['top1']:.3f} top5 {v['top5']:.3f}" for m, v in mean.items()))
(RESULTS_DIR / 'dev_test_scores.json').write_text(json.dumps({'per_subject': res, 'mean': mean}, indent=2))
"""),
    md("""
### Sanity check: shuffled labels must score at chance
We train again, but give every training recording a *random* video. If the score stays above
chance, something leaks (for example test data sneaking into training).
"""),
    code("""
from dataclasses import replace
shuf = summarise(score_dev_test(replace(cfg, shuffle_labels=True), feats))
print('shuffled: ' + '   '.join(f"{m} top1 {v['top1']:.3f} top5 {v['top5']:.3f}" for m, v in shuf.items()))
print('chance:   top1 0.010 top5 0.049')
"""),
]

NB05 = [
    md("""
# 05 · Predict and write the submission

Same idea as the organisers' notebook 02:
1. Train the chosen setting on **all training recordings** (all 6 subjects).
2. Save the trained models to `weights/` (sharing weights is required).
3. **Format check on sub-01**: write a ranked list exactly like the real submission and score
   it with sub-01's answers. If the number matches notebook 04, the format is right.
4. Write `submission/sub-05_ranked_candidate_list.npy` and `sub-06_…`: 102 × 102 integers.
   Row *i* = the *i*-th recording of `tst/sub-0X_tst.npy` in its **original order**. Each row
   holds the video ids 1001–1102, most likely first.

The sub-05/06 test recordings are used **only here**, once, to produce the predictions.
"""),
    code(SETUP),
    code("""
import joblib
from mindreading.pipeline import Config, Pipeline
from mindreading.video_features import extract_clip
from mindreading.config import TEST_IDS
from shared.ranking import scores_to_ranked_ids, check_ranked, evaluate, save_submission

cfg = Config(**json.loads((RESULTS_DIR / 'chosen_config.json').read_text()))
feats = extract_clip()
pipe = Pipeline(cfg, feats)
models = pipe.fit(None) if cfg.pool_subjects else {s: pipe.fit(s) for s in range(1, 7)}
WEIGHTS_DIR.mkdir(exist_ok=True)
joblib.dump({'config': cfg.__dict__, 'models': models}, WEIGHTS_DIR / 'base_ridge.joblib', compress=3)
print('saved', WEIGHTS_DIR / 'base_ridge.joblib')
"""),
    code("""
def ranked_for(sub):
    eeg, labels = pipe.test_data(sub)                  # original shuffled order, never re-sorted
    m = models if cfg.pool_subjects else models[sub]
    scores = pipe.scores(m, eeg, TEST_IDS)['combined' if len(cfg.methods) > 1 else cfg.methods[0]]
    return check_ranked(scores_to_ranked_ids(scores)), labels

r1, labels1 = ranked_for(1)
print('format check on sub-01:', r1.shape, r1.dtype, 'first row starts', r1[0, :5])
print('score with sub-01 answers:', evaluate(r1, labels1))
"""),
    code("""
SUBMISSION_DIR.mkdir(exist_ok=True)
for sub in (5, 6):
    ranked, _ = ranked_for(sub)
    path = save_submission(ranked, f'sub-{sub:02d}', SUBMISSION_DIR)
    back = check_ranked(np.load(path))
    print(path, back.shape, back.dtype, 'row 0 top-5:', back[0, :5])
"""),
]


def main():
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in [("03_video_features", NB03), ("04_train_and_validate", NB04),
                        ("05_predict_submission", NB05)]:
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        # Register once with: .venv\Scripts\python.exe -m ipykernel install --user --name hackathon311
        nb.metadata["kernelspec"] = {"name": "hackathon311", "display_name": "Python 3.11 (hackathon)", "language": "python"}
        nbf.write(nb, NB_DIR / f"{name}.ipynb")
        print("wrote", NB_DIR / f"{name}.ipynb")


if __name__ == "__main__":
    main()

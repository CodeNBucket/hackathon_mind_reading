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
# 03 · Stimulus features (the "cards")

The organisers' notebook 01 describes each video with **AlexNet** on 8 frames. We describe every
video (1,000 training + 102 test) with several complementary feature sets. Each was tested on
held-out training videos (see `results/experiment_log.md`):

| block | model | what it captures | used in final model |
|---|---|---|---|
| `clip` | CLIP ViT-B/32, 8 frames averaged | *what* is in the picture (objects, scenes) | yes (weight 1.0) |
| `mae` | VideoMAE (Kinetics), 16 frames, layers 4/8/12 | motion and actions | yes (0.6) |
| `low` | no model: brightness, contrast, saturation, motion, edges, scene cuts, loudness, sound onsets, every 0.1 s | low-level timing the early visual/auditory brain reacts to | yes (0.6) |
| `ast4` | Audio Spectrogram Transformer (AudioSet), layer 4 | what the **sound** is | yes (0.5) |

Everything is cached in `cache/`. The first run takes about 1 hour (most of it decoding video on
the CPU; models run on the GPU if there is one). Later runs load the cache in seconds.
"""),
    code(SETUP),
    code("""
from mindreading.video_features import extract_clip
from mindreading.features import EXTRACTORS, LOWLEVEL_NAMES
feats = {'clip_b32': extract_clip()}
for name in ('lowlevel', 'videomae', 'ast'):
    feats[name] = EXTRACTORS[name]()
for name, f in feats.items():
    print(f'{name:9s}', {k: v.shape for k, v in f.items() if k != 'ids'})
"""),
    md("""
### What the low-level time courses look like
Four videos, the 8 low-level signals over the 3 s (z-scored over all videos). The brain's first
responses (~0.1–0.2 s) follow exactly this kind of signal: changes in light, motion and sound.
"""),
    code("""
tc = feats['lowlevel']['timecourse']
z = (tc - tc.mean((0, 1))) / tc.std((0, 1))
t = np.arange(30) / 10
fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
for ax, row in zip(axes, [0, 10, 500, 1001]):
    ax.plot(t, z[row]); ax.set_title(f"video {feats['lowlevel']['ids'][row]}"); ax.set_xlabel('s')
axes[-1].legend(LOWLEVEL_NAMES, fontsize=7, loc='upper right'); plt.tight_layout(); plt.show()
print('videos without sound:', int(feats['ast']['silent'].sum()))
"""),
    md("""
### Sanity check: do similar videos get similar CLIP cards?
For three test videos: the most similar training videos (cosine similarity, 1 = identical).
"""),
    code("""
from mindreading.video_features import lookup
train_ids, test_ids = np.arange(1, 1001), np.arange(1001, 1103)
A, B = lookup(feats['clip_b32'], test_ids), lookup(feats['clip_b32'], train_ids)
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

Same idea as the organisers' notebook 01 (video features → ridge regression → predicted EEG →
compare with the real EEG), with the improvements we validated one by one.

**EEG preprocessing** (`mindreading/preprocess.py`)
1. 500 → 100 measurements per second (anti-alias filter, like the organisers).
2. **Channel whitening**: decorrelate the 17 electrodes using each file's own covariance (shrunk
   30% towards identity). The electrodes share a lot of noise; this removes it. It was the
   biggest single gain. It is unsupervised, so it is allowed on test files too.
3. z-score each file with its own statistics (the test files are scaled differently from the
   training files).
4. Keep 0–1 s after video onset (later windows only added noise).

**Model** (`mindreading/model.py`): for each subject, three encoders whose scores are added:
- **subject**: trained on that subject's own training recordings
- **pooled**: trained on all 6 subjects' training recordings
- **ga**: trained on the *grand average* EEG over subjects for each video (much cleaner target)

Each encoder: video features → ridge regression → predicted EEG, with a **separate
regularisation strength for every electrode × time point**, so unpredictable points shrink to
zero. A recording is compared with every candidate video by correlating the real EEG with the
EEG predicted for that video.
"""),
    code(SETUP),
    code("""
from mindreading.data import load_train
from mindreading.preprocess import downsample, normalise, whiten_channels
ids, raw = load_train(1)
x0 = normalise(downsample(raw))
x1 = normalise(whiten_channels(downsample(raw), 0.3))
t = np.arange(x0.shape[-1]) / 100
fig, ax = plt.subplots(1, 3, figsize=(16, 3.5))
ax[0].plot(t, x0.mean(0).T, lw=0.8); ax[0].set_title('sub-01 average response, before whitening')
ax[1].plot(t, x1.mean(0).T, lw=0.8); ax[1].set_title('after whitening')
c0 = np.corrcoef(x0.transpose(1, 0, 2).reshape(17, -1)); c1 = np.corrcoef(x1.transpose(1, 0, 2).reshape(17, -1))
ax[2].bar(['before', 'after'], [c0[np.triu_indices(17, 1)].mean(), c1[np.triu_indices(17, 1)].mean()])
ax[2].set_title('mean correlation between electrodes')
for a in ax[:2]: a.set_xlabel('seconds after video start')
plt.tight_layout(); plt.show()
"""),
    md("""
### How we got here
Every step was chosen on held-out *training* videos (never on test data). Full tables:
`results/experiment_log.md`.
"""),
    code("""
from IPython.display import Markdown, display
display(Markdown((RESULTS_DIR / 'experiment_log.md').read_text(encoding='utf-8')))
"""),
    md("""
### Validation on held-out training videos (3 random hidden sets, ~150 candidates, chance 3.3% top-5)
Takes a few minutes. Shows each ensemble member on its own and the full ensemble.
"""),
    code("""
from mindreading.model import Config, Model, validate, dev_test, summarise
cfg = Config()
model = Model(cfg)
val = validate(cfg, model=model)
for k, v in val.items():
    print(f"{k:9s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  median rank {v['median_rank']:.1f}")
"""),
    md("""
### Score on the sub-01…04 test sets (102 candidates, chance 1.0% / 4.9%)
Organisers' AlexNet baseline: about **4% top-1, 11.7% top-5**.
"""),
    code("""
res = dev_test(cfg, model=model)
for s, r in res.items():
    e = r['ensemble']
    print(f"sub-0{s}: top1 {e['top1']:.3f}  top5 {e['top5']:.3f}  median rank {e['median_rank']:.0f}")
mean = summarise(res)
for k, v in mean.items():
    print(f"mean {k:9s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / 'dev_test_scores.json').write_text(json.dumps(
    {'config': cfg.__dict__, 'validation': val, 'per_subject': res, 'mean': mean}, indent=2, default=str))
"""),
    md("""
### Sanity check: shuffled labels must score at chance
Train again, but pair every training recording with a *random* video. If the score stays above
chance, information is leaking from somewhere.
"""),
    code("""
from dataclasses import replace
shuf = summarise(dev_test(replace(cfg, shuffle_labels=True), model=Model(replace(cfg, shuffle_labels=True), model.feats)))
print(f"shuffled: top1 {shuf['ensemble']['top1']:.3f}  top5 {shuf['ensemble']['top5']:.3f}   (chance 0.010 / 0.049)")
"""),
]

NB05 = [
    md("""
# 05 · Predict and write the submission

Same idea as the organisers' notebook 02:
1. Train the final model on **all training recordings** of all 6 subjects (the rules allow
   sub-05/06's *training* files).
2. Save the trained encoders to `weights/` (sharing weights is required).
3. **Format check on sub-01**: build the ranked list exactly like the real submission and score
   it with sub-01's answers. It must match notebook 04.
4. Write `submission/sub-05_ranked_candidate_list.npy` and `sub-06_…`: 102 × 102 integers. Row *i*
   = the *i*-th recording of `tst/sub-0X_tst.npy` in its **original order**. Each row lists the
   video ids 1001–1102, most likely first.

The sub-05/06 test recordings are used **only here**, once, to produce the predictions.
"""),
    code(SETUP),
    code("""
import joblib
from mindreading.model import Config, Model, predict_scores
from mindreading.ranking import scores_to_ranked_ids, check_ranked, evaluate, save_submission

cfg = Config()
model = Model(cfg).fit_shared()
members = {s: {**model.fit_subject(s), **model.shared} for s in (1, 5, 6)}
WEIGHTS_DIR.mkdir(exist_ok=True)
weights = {'config': cfg.__dict__,
           'shared': {k: {'mu': m.mu, 'coef': m.ridge.coef_.astype(np.float32), 'intercept': m.ridge.intercept_}
                      for k, m in model.shared.items()},
           'subject': {s: {'mu': m['subject'].mu, 'coef': m['subject'].ridge.coef_.astype(np.float32),
                           'intercept': m['subject'].ridge.intercept_} for s, m in members.items()}}
joblib.dump(weights, WEIGHTS_DIR / 'ensemble_encoders.joblib', compress=3)
print('saved', WEIGHTS_DIR / 'ensemble_encoders.joblib')
"""),
    code("""
r1 = check_ranked(scores_to_ranked_ids(predict_scores(model, 1, members[1])))
labels1 = np.load(DATA_DIR / 'tst_stim' / 'sub-01_tst_stim.npy')
print('format check on sub-01:', r1.shape, r1.dtype, 'first row starts', r1[0, :5])
print('score with sub-01 answers:', evaluate(r1, labels1))
"""),
    code("""
SUBMISSION_DIR.mkdir(exist_ok=True)
for sub in (5, 6):
    ranked = check_ranked(scores_to_ranked_ids(predict_scores(model, sub, members[sub])))
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

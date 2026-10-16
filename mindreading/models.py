"""The two linear models and how they score candidates.

Encoding (organisers' direction): video features -> ridge -> predicted EEG.
    Score(recording, video) = Pearson correlation between the real EEG and the EEG
    predicted for that video.
Decoding (our direction): EEG -> ridge -> predicted video features.
    Score(recording, video) = cosine similarity between the predicted and the real
    video features.
Combined: z-score each recording's scores over the candidates and add the two.

Every score matrix is (n_recordings, n_candidates): higher = more likely.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 6, 15)


def _rowz(a):
    a = a - a.mean(axis=1, keepdims=True)
    return a / (a.std(axis=1, keepdims=True) + 1e-8)


def correlation_matrix(a, b):
    """Pearson correlation between every row of a and every row of b."""
    return _rowz(a) @ _rowz(b).T / a.shape[1]


def cosine_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


class Encoder:
    """Video features -> EEG (n, 17, T). Same idea as the organisers' baseline, with CLIP."""

    def __init__(self, n_pca=None):
        steps = [StandardScaler()]
        if n_pca:
            steps += [PCA(n_pca, random_state=0), StandardScaler()]
        steps += [RidgeCV(alphas=ALPHAS)]
        self.model = make_pipeline(*steps)

    def fit(self, video_feats, eeg):
        self.shape = eeg.shape[1:]
        self.model.fit(video_feats, eeg.reshape(len(eeg), -1))
        return self

    def predict_eeg(self, video_feats):
        return self.model.predict(video_feats).reshape(len(video_feats), *self.shape)

    def scores(self, eeg, candidate_feats):
        pred = self.predict_eeg(candidate_feats)
        return correlation_matrix(eeg.reshape(len(eeg), -1), pred.reshape(len(pred), -1))


class Decoder:
    """EEG (n, 17, T) -> video features. Our direction: brain -> 'card'."""

    def __init__(self, n_pca=None):
        steps = [StandardScaler()]
        if n_pca:
            steps += [PCA(n_pca, random_state=0), StandardScaler()]
        steps += [RidgeCV(alphas=ALPHAS)]
        self.model = make_pipeline(*steps)

    def fit(self, eeg, video_feats):
        self.model.fit(eeg.reshape(len(eeg), -1), video_feats)
        return self

    def predict_feats(self, eeg):
        return self.model.predict(eeg.reshape(len(eeg), -1))

    def scores(self, eeg, candidate_feats):
        return cosine_matrix(self.predict_feats(eeg), candidate_feats)


def combine(*score_mats, weights=None):
    weights = weights or [1.0] * len(score_mats)
    return sum(w * _rowz(s) for w, s in zip(weights, score_mats))


def rank_of_true(scores, true_cols):
    """1-based rank of the true candidate in each row (1 = best)."""
    true_score = scores[np.arange(len(scores)), true_cols]
    return (scores > true_score[:, None]).sum(axis=1) + 1


def topk(scores, true_cols):
    r = rank_of_true(scores, true_cols)
    return {"top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "median_rank": float(np.median(r))}

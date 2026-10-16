"""Turn score matrices into the organizers' submission format, and score them.

Conventions (from the brief):
- recordings are in the original order of tst/subXX_tst.npy (never re-sorted)
- candidate videos are 1001..1102 (stimuli_test), column j <-> candidate_ids[j]
- a submission is an int array (n_recordings, n_candidates); row i lists candidate
  video IDS (not column indices) from most to least likely
"""
import numpy as np

CANDIDATE_IDS = np.arange(1001, 1103)  # 102 test videos


def scores_to_ranked_ids(scores, candidate_ids=CANDIDATE_IDS):
    """scores: (n_recordings, n_candidates), higher = more likely. Returns ranked video IDs."""
    scores = np.asarray(scores)
    candidate_ids = np.asarray(candidate_ids)
    if scores.ndim != 2 or scores.shape[1] != len(candidate_ids):
        raise ValueError(f"scores must be (n_recordings, {len(candidate_ids)}), got {scores.shape}")
    order = np.argsort(-scores, axis=1, kind="stable")
    return candidate_ids[order].astype(np.int64)


def check_ranked(ranked, candidate_ids=CANDIDATE_IDS):
    """Raise if `ranked` is not a valid submission array. Returns it unchanged."""
    ranked = np.asarray(ranked)
    n = len(candidate_ids)
    if ranked.shape != (n, n):
        raise ValueError(f"expected shape ({n}, {n}), got {ranked.shape}")
    if not np.issubdtype(ranked.dtype, np.integer):
        raise ValueError(f"expected an integer array, got {ranked.dtype}")
    want = np.sort(candidate_ids)
    if not (np.sort(ranked, axis=1) == want).all():
        raise ValueError("every row must contain each candidate video id exactly once (ids, not column indices)")
    return ranked


def top_k_accuracy(ranked, true_ids, k):
    """Fraction of recordings whose true video is among the first k entries of its row."""
    ranked = np.asarray(ranked)
    true_ids = np.asarray(true_ids)
    return float((ranked[:, :k] == true_ids[:, None]).any(axis=1).mean())


def evaluate(ranked, true_ids):
    return {"top1": top_k_accuracy(ranked, true_ids, 1), "top5": top_k_accuracy(ranked, true_ids, 5)}


def save_submission(ranked, subject, out_dir="."):
    """Validate and write <subject>_ranked_candidate_list.npy (subject like 'sub-05')."""
    check_ranked(ranked)
    path = f"{out_dir}/{subject}_ranked_candidate_list.npy"
    np.save(path, np.asarray(ranked, dtype=np.int64))
    return path


if __name__ == "__main__":  # self-test on synthetic data
    rng = np.random.default_rng(0)
    n = len(CANDIDATE_IDS)
    true = rng.permutation(CANDIDATE_IDS)                    # shuffled test order, like tst_stim
    perfect = np.where(true[:, None] == CANDIDATE_IDS[None, :], 1.0, 0.0) + rng.normal(0, 1e-3, (n, n))
    r = check_ranked(scores_to_ranked_ids(perfect))
    assert evaluate(r, true) == {"top1": 1.0, "top5": 1.0}, "perfect scores must give 100%"
    chance = evaluate(scores_to_ranked_ids(rng.normal(size=(n, n))), true)
    assert chance["top1"] < 0.1 and chance["top5"] < 0.2, chance
    for bad in (r[:, :-1], r - 1001, r.astype(float)):       # wrong shape, column indices, float dtype
        try:
            check_ranked(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad submission accepted")
    print("ok: perfect=100%, random top1/top5 =", chance)

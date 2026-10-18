"""Task 4: replace / complement the linear ridge encoder (video features -> EEG wave).

Variants (only the EEG-wave encoders change; the rhythm-power encoder stays ridge):
  ridge        current model (baseline)
  kernel       kernel ridge regression, RBF kernel (non-linear, convex, rarely overfits)
  resmlp       ridge + a small neural network trained on ridge's residuals (MSE)
  contrastive  neural network trained with an InfoNCE loss on the correlation between predicted
               and real EEG: "your prediction must match the right recording better than the
               other recordings in the batch" (the CLIP training idea, applied to our scoring)
Plus ensembles of each variant with ridge. Everything on held-out training videos (3 splits).

    .venv\\Scripts\\python.exe past_trials\\task4_encoders.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mindreading.model as M  # noqa: E402
from mindreading.config import SEED, SUBJECTS  # noqa: E402
from mindreading.model import Config, Model, _rowz, split_train_videos, topk  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WAVE_DIM = 17 * 100


# ---------------------------------------------------------------- encoders

class KernelEnc:
    def __init__(self, alpha=1.0, gamma_scale=1.0):
        self.alpha, self.gamma_scale = alpha, gamma_scale

    def fit(self, V, Y, ids=None):
        self.mu = V.mean(0)
        X = V - self.mu
        rng = np.random.default_rng(SEED)
        s = X[rng.choice(len(X), min(500, len(X)), replace=False)]
        sq = (s ** 2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2 * s @ s.T          # squared distances without a 3-D array
        gamma = self.gamma_scale / np.median(d2[d2 > 0])
        self.ym = Y.mean(0)
        self.kr = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma).fit(X, Y - self.ym)
        return self

    def predict(self, V):
        return self.kr.predict(V - self.mu) + self.ym


class Net(nn.Module):
    def __init__(self, d_in, d_out, hidden=1024, dropout=0.5):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.lin(x) + self.mlp(x)


def _corr_rows(a, b):
    a = a - a.mean(1, keepdim=True); a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    b = b - b.mean(1, keepdim=True); b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    return a @ b.T


class TorchEnc:
    """mode='resmlp': ridge + net on residuals (MSE). mode='contrastive': net trained with InfoNCE
    (+ small MSE term) on correlation between predicted and real EEG. Early stopping on 10% of the
    training videos (held out inside the fit set, never the validation videos)."""

    def __init__(self, mode, epochs=150, lr=1e-3, wd=5e-2, hidden=1024, dropout=0.5, tau=0.05, batch=256):
        self.mode, self.epochs, self.lr, self.wd = mode, epochs, lr, wd
        self.hidden, self.dropout, self.tau, self.batch = hidden, dropout, tau, batch

    def fit(self, V, Y, ids):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        self.mu = V.mean(0)
        X = (V - self.mu).astype(np.float32)
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
        T = ((Y - self.ym) / self.ys).astype(np.float32)
        if self.mode == "resmlp":
            self.ridge = M.Encoder().fit(V, Y)
            T = ((Y - self.ridge.predict(V)) / self.ys).astype(np.float32)
        uniq = np.unique(ids)
        es_vid = rng.choice(uniq, max(1, len(uniq) // 10), replace=False)
        es = np.isin(ids, es_vid)
        Xt, Tt, It = (torch.tensor(a[~es]).to(DEV) for a in (X, T, ids.astype(np.int64)))
        Xe, Te = torch.tensor(X[es]).to(DEV), torch.tensor(T[es]).to(DEV)
        Ye_full = torch.tensor(((Y - self.ym) / self.ys)[es].astype(np.float32)).to(DEV)
        self.net = Net(X.shape[1], T.shape[1], self.hidden, self.dropout).to(DEV)
        if self.mode == "contrastive":   # start from the ridge solution: the net only has to improve it
            r = M.Encoder().fit(V, (Y - self.ym) / self.ys).ridge
            with torch.no_grad():
                self.net.lin.weight.copy_(torch.tensor(r.coef_, dtype=torch.float32))
                self.net.lin.bias.copy_(torch.tensor(r.intercept_, dtype=torch.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = -np.inf, None, 0
        n = len(Xt)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=DEV)
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                pred = self.net(Xt[b])
                if self.mode == "resmlp":
                    loss = ((pred - Tt[b]) ** 2).mean()
                else:
                    sim = _corr_rows(Tt[b], pred) / self.tau          # real i vs predicted j
                    same = It[b][:, None] == It[b][None, :]
                    logits = sim.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool, device=DEV), -1e4)
                    tgt = torch.arange(len(b), device=DEV)
                    loss = (nn.functional.cross_entropy(logits, tgt) + nn.functional.cross_entropy(logits.T, tgt)) / 2
                    loss = loss + 0.1 * ((pred - Tt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            # early stopping: ranking quality on the held-out 10% (top-5 among those videos)
            self.net.eval()
            with torch.no_grad():
                pe = self.net(Xe)
                if self.mode == "resmlp":
                    pe = pe + torch.tensor(((self.ridge.predict(V[es]) - self.ym) / self.ys).astype(np.float32), device=DEV)
                sim = _corr_rows(Ye_full, pe)
                r = (sim > sim.diag()[:, None]).sum(1) + 1
                metric = (r <= 5).float().mean().item() - 1e-4 * r.float().mean().item()
            if metric > best:
                best, bad = metric, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 15:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.epochs_used = ep + 1
        return self

    def predict(self, V):
        with torch.no_grad():
            p = self.net(torch.tensor((V - self.mu).astype(np.float32), device=DEV)).cpu().numpy()
        p = p * self.ys
        if self.mode == "resmlp":
            return self.ridge.predict(V) + p
        return p + self.ym


# ---------------------------------------------------------------- evaluation

def make_model(factory):
    model = Model(Config())

    def enc(self, ids, Y, _f=factory):
        if self.cfg.shuffle_labels:
            ids = np.random.default_rng(SEED).permutation(ids)
        V = self.feats(ids)
        if Y.shape[1] == WAVE_DIM and _f is not None:
            return _f().fit(V, Y, np.asarray(ids))
        return M.Encoder().fit(V, Y)

    model._encoder = enc.__get__(model)
    return model


def val_scores(model):
    """{(seed, sub): (ensemble score matrix, true columns)} on held-out training videos."""
    out = {}
    for seed in (0, 1, 2):
        fit_ids, val_ids = split_train_videos(seed=seed)
        model.fit_shared(fit_ids)
        for s in SUBJECTS:
            members = {**model.fit_subject(s, fit_ids), **model.shared}
            ids, Y = model.train_eeg(s)
            m = np.isin(ids, val_ids)
            cand = np.unique(ids[m])
            sc = model.scores(members, Y[m], cand, model.train_power(s)[m])
            out[(seed, s)] = (sc["ensemble"], np.searchsorted(cand, ids[m]))
    return out


def summary(scores):
    r = [topk(S, t) for S, t in scores.values()]
    return {k: float(np.mean([x[k] for x in r])) for k in ("top1", "top5", "median_rank")}


VARIANTS = {
    "ridge (current)": None,
    "kernel ridge a=1": lambda: KernelEnc(alpha=1.0),
    "kernel ridge a=0.1": lambda: KernelEnc(alpha=0.1),
    "ridge + residual MLP": lambda: TorchEnc("resmlp"),
    "contrastive MLP": lambda: TorchEnc("contrastive"),
}

if __name__ == "__main__":
    results, cache = {}, {}
    for name, factory in VARIANTS.items():
        t0 = time.time()
        cache[name] = val_scores(make_model(factory))
        results[name] = summary(cache[name])
        v = results[name]
        print(f"{name:28s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  median {v['median_rank']:.1f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    base = cache["ridge (current)"]
    for name in VARIANTS:
        if name == "ridge (current)":
            continue
        for w in (0.5, 1.0):
            comb = {k: (_rowz(base[k][0]) + w * _rowz(cache[name][k][0]), base[k][1]) for k in base}
            key = f"ridge + {w} x {name}"
            results[key] = summary(comb)
            v = results[key]
            print(f"{key:28s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  median {v['median_rank']:.1f}", flush=True)
    (Path(__file__).resolve().parent / "task4_encoders_result.json").write_text(json.dumps(results, indent=2))

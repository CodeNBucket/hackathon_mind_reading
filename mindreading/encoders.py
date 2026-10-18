"""Contrastive neural-network encoder: video features -> EEG wave, trained for ranking.

The ridge encoder (model.py) is fitted to predict the EEG as exactly as possible. This network
is trained for what we are scored on: its predicted EEG must correlate better with the right
recording than with the other recordings in the batch (InfoNCE loss, the idea CLIP is trained
with), plus a small MSE term. It starts from the ridge solution (linear path) and adds a small
non-linear path. Early stopping uses 10% of the *training* videos held out inside the fit set.
Three networks with different seeds are averaged, because neural networks vary run to run.
"""
import numpy as np

from .config import SEED


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveEncoder:
    def __init__(self, seed=SEED, epochs=150, lr=1e-3, wd=5e-2, hidden=1024, dropout=0.5, tau=0.05,
                 batch=256, patience=15, mse_weight=0.1):
        self.seed, self.epochs, self.lr, self.wd = seed, epochs, lr, wd
        self.hidden, self.dropout, self.tau, self.batch = hidden, dropout, tau, batch
        self.patience, self.mse_weight = patience, mse_weight

    def _net(self, d_in, d_out):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lin = nn.Linear(d_in, d_out)                                 # linear path (ridge init)
                s.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, self.hidden), nn.GELU(),
                                      nn.Dropout(self.dropout), nn.Linear(self.hidden, d_out))

            def forward(s, x):
                return s.lin(x) + s.mlp(x)

        return Net()

    def fit(self, V, Y, ids):
        import torch
        import torch.nn.functional as F
        from .model import Encoder                                               # ridge, for the start point
        dev = _device()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu = V.mean(0)
        X = (V - self.mu).astype(np.float32)
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
        T = ((Y - self.ym) / self.ys).astype(np.float32)
        uniq = np.unique(ids)
        es = np.isin(ids, rng.choice(uniq, max(1, len(uniq) // 10), replace=False))
        Xt, Tt = torch.tensor(X[~es], device=dev), torch.tensor(T[~es], device=dev)
        It = torch.tensor(np.asarray(ids)[~es].astype(np.int64), device=dev)
        Xe, Te = torch.tensor(X[es], device=dev), torch.tensor(T[es], device=dev)

        self.net = self._net(X.shape[1], T.shape[1]).to(dev)
        r = Encoder().fit(V, T).ridge
        with torch.no_grad():
            self.net.lin.weight.copy_(torch.tensor(r.coef_, dtype=torch.float32))
            self.net.lin.bias.copy_(torch.tensor(r.intercept_, dtype=torch.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        def corr(a, b):
            a = a - a.mean(1, keepdim=True); a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
            b = b - b.mean(1, keepdim=True); b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
            return a @ b.T

        best, best_state, bad = -np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), device=dev)
            for s in range(0, len(Xt), self.batch):
                b = perm[s:s + self.batch]
                pred = self.net(Xt[b])
                sim = corr(Tt[b], pred) / self.tau
                same = It[b][:, None] == It[b][None, :]                         # same video = not a negative
                sim = sim.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool, device=dev), -1e4)
                tgt = torch.arange(len(b), device=dev)
                loss = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt)) / 2
                loss = loss + self.mse_weight * ((pred - Tt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                sim = corr(Te, self.net(Xe))
                rank = (sim > sim.diag()[:, None]).sum(1) + 1
                metric = (rank <= 5).float().mean().item() - 1e-4 * rank.float().mean().item()
            if metric > best:
                best, bad = metric, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, V):
        import torch
        with torch.no_grad():
            x = torch.tensor((V - self.mu).astype(np.float32), device=next(self.net.parameters()).device)
            return self.net(x).cpu().numpy() * self.ys + self.ym

    def state(self):
        """Everything needed to rebuild the network (for weights/)."""
        return {"mu": self.mu, "ym": self.ym, "ys": self.ys,                    # float16 keeps files < 100 MB
                "net": {k: v.cpu().numpy().astype(np.float16) for k, v in self.net.state_dict().items()}}


class SeedAverage:
    """Average of n ContrastiveEncoders trained with seeds SEED, SEED+1, ..."""

    def __init__(self, n=3, **kw):
        self.n, self.kw = n, kw

    def fit(self, V, Y, ids):
        self.encoders = [ContrastiveEncoder(seed=SEED + k, **self.kw).fit(V, Y, ids) for k in range(self.n)]
        return self

    def predict(self, V):
        return np.mean([e.predict(V) for e in self.encoders], axis=0)

    def state(self):
        return [e.state() for e in self.encoders]

"""Ablation: why the ridge-initialised contrastive NN? Same splits as rounds 14/15 (seeds 0-2).

  A  ridge start + InfoNCE (current)
  B  random start + InfoNCE          (no ridge start)
  C  random start + MSE only         (plain NN predicting the EEG)
All with 1 network (no seed averaging) so only the ingredient changes.

    .venv\Scripts\python.exe past_trials\task7_nn_ablation.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mindreading.encoders as enc
from mindreading.model import Config, Model, VideoFeatures, validate


class Variant(enc.ContrastiveEncoder):
    ridge_init, use_infonce = True, True

    def fit(self, V, Y, ids):
        import torch
        import torch.nn.functional as F
        from mindreading.model import Encoder
        dev = enc._device()
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
        if self.ridge_init:
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
                mse = ((pred - Tt[b]) ** 2).mean()
                if self.use_infonce:
                    sim = corr(Tt[b], pred) / self.tau
                    same = It[b][:, None] == It[b][None, :]
                    sim = sim.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool, device=dev), -1e4)
                    tgt = torch.arange(len(b), device=dev)
                    loss = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt)) / 2 + self.mse_weight * mse
                else:
                    loss = mse
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


cfg = Config(blocks={"clip": 1.0, "ast4": 0.5, "low": 0.6, "mae": 1.5}, contrastive_seeds=1, colnorm=False)  # as run (round 17)
feats = VideoFeatures(cfg.blocks)
for tag, init, nce in [("A ridge start + InfoNCE (current)", True, True),
                       ("B random start + InfoNCE", False, True),
                       ("C plain NN: random start + MSE", False, False)]:
    Variant.ridge_init, Variant.use_infonce = init, nce
    enc.ContrastiveEncoder = Variant
    t0 = time.time()
    v = validate(cfg, seeds=(0, 1, 2), model=Model(cfg, feats))
    c, e, r = v["wave_c"], v["ensemble"], v["wave"]
    print(f"{tag:36s} NN alone top1 {c['top1']:.3f} top5 {c['top5']:.3f} median {c['median_rank']:.1f} | "
          f"ridge + 2xNN top1 {e['top1']:.3f} top5 {e['top5']:.3f} median {e['median_rank']:.1f} | "
          f"ridge alone top5 {r['top5']:.3f}  ({time.time() - t0:.0f} s)", flush=True)

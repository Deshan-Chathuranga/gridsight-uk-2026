"""TCN-Q base learner: a Temporal Convolutional Network with a quantile head.

Captures the ~63h diurnal context from a sequence of past half-hourly features
(dilated *causal* convolutions => no peeking into the future). One forward pass
outputs all quantiles; trained with the summed pinball loss. Quantiles are sorted
at predict time so the base output itself never crosses.

Architecture: Bai et al. (2018) TCN — stacked residual blocks of two dilated
causal Conv1d layers, dilation doubling each block so the receptive field covers
the full sequence length.
"""
from __future__ import annotations

import numpy as np


def device() -> str:
    """Pick the best available device: CUDA (Colab GPU) -> MPS (Apple) -> CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_torch(cfg, n_features: int, n_quantiles: int):
    import torch
    import torch.nn as nn

    class Chomp1d(nn.Module):
        def __init__(self, s): super().__init__(); self.s = s
        def forward(self, x): return x[:, :, :-self.s].contiguous() if self.s else x

    class TemporalBlock(nn.Module):
        def __init__(self, ci, co, k, d, dropout):
            super().__init__()
            pad = (k - 1) * d
            self.net = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ci, co, k, padding=pad, dilation=d)),
                Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Conv1d(co, co, k, padding=pad, dilation=d)),
                Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            )
            self.down = nn.Conv1d(ci, co, 1) if ci != co else None
            self.relu = nn.ReLU()
        def forward(self, x):
            out = self.net(x)
            res = x if self.down is None else self.down(x)
            return self.relu(out + res)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ci = [], n_features
            for i, co in enumerate(cfg.tcn_channels):
                layers.append(TemporalBlock(ci, co, cfg.tcn_kernel, 2 ** i, cfg.tcn_dropout))
                ci = co
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(ci, n_quantiles)
        def forward(self, x):                      # x: (B, L, F)
            h = self.tcn(x.transpose(1, 2))        # -> (B, C, L)
            return self.head(h[:, :, -1])          # last step -> (B, Q)

    return TCN()


def pinball_loss_torch(pred, target, quantiles):
    import torch
    target = target.unsqueeze(1)                   # (B,1)
    q = torch.tensor(quantiles, device=pred.device).view(1, -1)
    d = target - pred
    return torch.mean(torch.maximum(q * d, (q - 1.0) * d))


class TCNQuantile:
    def __init__(self, cfg, n_features: int):
        self.cfg = cfg
        self.quantiles = tuple(cfg.quantiles)
        self.n_features = n_features
        self.model_ = None

    def build(self):
        """Construct the (untrained) network — used when reloading saved weights."""
        import torch
        torch.set_num_threads(self.cfg.torch_threads)
        self.model_ = _build_torch(self.cfg, self.n_features, len(self.quantiles))
        return self

    def fit(self, seqs: np.ndarray, y: np.ndarray,
            val: tuple | None = None) -> "TCNQuantile":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from loguru import logger
        torch.set_num_threads(self.cfg.torch_threads)   # prevent OpenMP oversubscription
        torch.manual_seed(self.cfg.seed)
        dev = device()                                   # cuda (Colab) -> mps (Mac) -> cpu
        self.model_ = _build_torch(self.cfg, self.n_features, len(self.quantiles)).to(dev)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.cfg.tcn_lr)

        ds = TensorDataset(torch.from_numpy(seqs), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.cfg.tcn_batch, shuffle=True, drop_last=False)
        E = self.cfg.tcn_epochs
        for ep in range(E):
            self.model_.train()
            tot, nb = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = pinball_loss_torch(self.model_(xb), yb, self.quantiles)
                loss.backward()
                opt.step()
                tot += float(loss.item()); nb += 1
            if ep == 0 or (ep + 1) % max(1, E // 5) == 0 or ep == E - 1:
                logger.info(f"    TCN[{dev}] epoch {ep+1}/{E} · pinball={tot/max(nb,1):.4f}")
        return self

    def predict(self, seqs: np.ndarray) -> dict[float, np.ndarray]:
        import torch
        torch.set_num_threads(self.cfg.torch_threads)
        dev = next(self.model_.parameters()).device
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(seqs), 1024):
                xb = torch.from_numpy(seqs[i:i + 1024]).to(dev)
                outs.append(self.model_(xb).cpu().numpy())
        pred = np.concatenate(outs, axis=0) if outs else np.empty((0, len(self.quantiles)))
        pred = np.sort(pred, axis=1)               # enforce monotone quantiles
        return {q: pred[:, i].astype("float32") for i, q in enumerate(self.quantiles)}

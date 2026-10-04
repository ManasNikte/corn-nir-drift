"""A small 1-D CNN for spectra, with a scikit-learn style fit/predict interface."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class SpectraNet(nn.Module):
    """Three conv/pool blocks followed by a small dense head.

    Spectral features are position-specific (a peak at 1940 nm means water), so we
    flatten rather than global-pool, keeping wavelength information for the head.
    """

    def __init__(
        self,
        n_channels: int,
        n_outputs: int,
        width: int = 8,
        hidden: int = 32,
        kernel: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = kernel // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, width, kernel, padding=pad),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(width, 2 * width, kernel, padding=pad),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(2 * width, 2 * width, kernel, padding=pad),
            nn.ELU(),
            nn.AvgPool1d(2),
        )
        n_flat = 2 * width * (n_channels // 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_flat, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1)))


def augment(x: torch.Tensor, strength: float, gen: torch.Generator) -> torch.Tensor:
    """Random multiplicative scale, offset, linear tilt and noise (in standardised units)."""
    n, L = x.shape
    t = torch.linspace(-1, 1, L)
    r = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    scale = 1 + 0.05 * strength * r(n, 1)
    offset = 0.1 * strength * r(n, 1)
    tilt = 0.1 * strength * r(n, 1) * t
    noise = 0.02 * strength * r(n, L)
    return x * scale + offset + tilt + noise


class CNNRegressor(BaseEstimator, RegressorMixin):
    """Multi-output CNN regressor with input/target standardisation and early stopping."""

    def __init__(
        self,
        epochs: int = 400,
        lr: float = 2e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 16,
        val_fraction: float = 0.15,
        patience: int = 60,
        width: int = 8,
        hidden: int = 32,
        dropout: float = 0.2,
        augment_strength: float = 0.0,
        n_ensemble: int = 1,
        random_state: int = 0,
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patience = patience
        self.width = width
        self.hidden = hidden
        self.dropout = dropout
        self.augment_strength = augment_strength
        self.n_ensemble = n_ensemble
        self.random_state = random_state

    def _fit_one(self, X: torch.Tensor, Y: torch.Tensor, seed: int) -> nn.Module:
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        n = len(X)
        perm = torch.randperm(n, generator=gen)
        n_val = max(1, int(round(self.val_fraction * n))) if self.val_fraction > 0 else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        net = SpectraNet(X.shape[1], Y.shape[1], self.width, self.hidden, dropout=self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best, best_state, stale = np.inf, copy.deepcopy(net.state_dict()), 0

        for _ in range(self.epochs):
            net.train()
            order = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
            for start in range(0, len(order), self.batch_size):
                b = order[start : start + self.batch_size]
                xb = X[b]
                if self.augment_strength > 0:
                    xb = augment(xb, self.augment_strength, gen)
                opt.zero_grad()
                loss_fn(net(xb), Y[b]).backward()
                opt.step()
            if n_val:
                net.eval()
                with torch.no_grad():
                    val = loss_fn(net(X[val_idx]), Y[val_idx]).item()
                if val < best - 1e-5:
                    best, best_state, stale = val, copy.deepcopy(net.state_dict()), 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if n_val:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        self._1d = Y.ndim == 1
        Y = Y.reshape(len(Y), -1)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-8
        self.y_mean_, self.y_std_ = Y.mean(0), Y.std(0) + 1e-8
        Xt = torch.from_numpy((X - self.x_mean_) / self.x_std_)
        Yt = torch.from_numpy((Y - self.y_mean_) / self.y_std_)
        self.nets_ = [self._fit_one(Xt, Yt, self.random_state + i) for i in range(self.n_ensemble)]
        return self

    def predict(self, X):
        Xt = torch.from_numpy((np.asarray(X, dtype=np.float32) - self.x_mean_) / self.x_std_)
        with torch.no_grad():
            pred = torch.stack([net(Xt) for net in self.nets_]).mean(0).numpy()
        pred = pred * self.y_std_ + self.y_mean_
        return pred.ravel().astype(float) if self._1d else pred.astype(float)

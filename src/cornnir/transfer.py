"""Calibration transfer: map spectra from a "slave" instrument onto the "master".

All methods are fitted on a small set of transfer standards measured on both
instruments, so the master calibration model can be reused unchanged.
"""

from __future__ import annotations

import numpy as np


def _truncated_pinv(A: np.ndarray, rank: int | None, rcond: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s[0] if s.size and s[0] > 0 else np.zeros_like(s, dtype=bool)
    if rank is not None:
        keep[rank:] = False
    return (Vt[keep].T / s[keep]) @ U[:, keep].T


class DirectStandardization:
    """DS (Wang, Veltkamp & Kowalski 1991): one global transfer matrix x_m = x_s F + b."""

    def __init__(self, rank: int | None = None, rcond: float = 1e-3):
        self.rank = rank
        self.rcond = rcond

    def fit(self, X_master: np.ndarray, X_slave: np.ndarray):
        mm, ms = X_master.mean(0), X_slave.mean(0)
        self.F_ = _truncated_pinv(X_slave - ms, self.rank, self.rcond) @ (X_master - mm)
        self.b_ = mm - ms @ self.F_
        return self

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        return X_slave @ self.F_ + self.b_


class PiecewiseDirectStandardization:
    """PDS (Wang et al. 1991): each master channel is regressed on a local slave window.

    The result is a banded transfer matrix, which needs far fewer standards than DS
    because each local regression has only ``2*half_window+1`` inputs.
    """

    def __init__(self, half_window: int = 5, rank: int | None = None, rcond: float = 1e-2):
        self.half_window = half_window
        self.rank = rank
        self.rcond = rcond

    def fit(self, X_master: np.ndarray, X_slave: np.ndarray):
        X_master = np.asarray(X_master, float)
        X_slave = np.asarray(X_slave, float)
        if X_master.shape != X_slave.shape:
            raise ValueError("master and slave standards must have the same shape")
        n_ch = X_master.shape[1]
        mm, ms = X_master.mean(0), X_slave.mean(0)
        Mc, Sc = X_master - mm, X_slave - ms
        F = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            lo, hi = max(0, i - self.half_window), min(n_ch, i + self.half_window + 1)
            F[lo:hi, i] = _truncated_pinv(Sc[:, lo:hi], self.rank, self.rcond) @ Mc[:, i]
        self.F_ = F
        self.b_ = mm - ms @ F
        return self

    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        return np.asarray(X_slave, float) @ self.F_ + self.b_


class SlopeBiasCorrection:
    """Correct *predictions* rather than spectra: y_corrected = a + b * y_pred, per target."""

    def fit(self, y_pred_slave: np.ndarray, y_true: np.ndarray):
        P = np.asarray(y_pred_slave, float).reshape(len(y_pred_slave), -1)
        Y = np.asarray(y_true, float).reshape(len(y_true), -1)
        self.slope_ = np.array([np.polyfit(P[:, j], Y[:, j], 1)[0] for j in range(P.shape[1])])
        self.bias_ = Y.mean(0) - self.slope_ * P.mean(0)
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        P = np.asarray(y_pred, float)
        return self.bias_ + self.slope_ * P.reshape(len(P), -1)

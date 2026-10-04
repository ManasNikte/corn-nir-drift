"""Spectral preprocessing in plain NumPy, wrapped as scikit-learn transformers.

All transformers operate row-wise on a 2-D array of spectra (n_samples, n_channels),
so they can be chained with ``sklearn.pipeline.Pipeline`` and cross-validated
without leakage (only MSC learns anything from the training data).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class _Stateless(TransformerMixin, BaseEstimator):
    """Row-wise transform that learns nothing from the training data."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Savitzky-Golay
# --------------------------------------------------------------------------- #


def _poly_deriv_matrix(t: np.ndarray, polyorder: int, deriv: int) -> np.ndarray:
    """Rows evaluate the ``deriv``-th derivative of sum_m a_m t^m at each t."""
    D = np.zeros((t.size, polyorder + 1))
    for m in range(deriv, polyorder + 1):
        D[:, m] = factorial(m) / factorial(m - deriv) * t ** (m - deriv)
    return D


def savgol_coeffs(window: int, polyorder: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """Convolution weights for the centre point of a Savitzky-Golay window."""
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3")
    if not 0 <= deriv <= polyorder < window:
        raise ValueError("require 0 <= deriv <= polyorder < window")
    half = window // 2
    t = np.arange(-half, half + 1, dtype=float)
    V = np.vander(t, polyorder + 1, increasing=True)
    # least-squares polynomial coefficients are pinv(V) @ x; derivative at t=0 is m! * a_m
    return factorial(deriv) * np.linalg.pinv(V)[deriv] / delta**deriv


def savgol(
    X: np.ndarray, window: int, polyorder: int, deriv: int = 0, delta: float = 1.0
) -> np.ndarray:
    """Savitzky-Golay smoothing/differentiation along the last axis.

    Edges are handled by fitting the polynomial to the first/last full window and
    evaluating it at the edge points (equivalent to scipy's ``mode='interp'``).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_ch = X.shape[1]
    if window > n_ch:
        raise ValueError("window longer than the spectrum")
    half = window // 2
    c = savgol_coeffs(window, polyorder, deriv, delta)

    out = np.empty_like(X)
    windows = np.lib.stride_tricks.sliding_window_view(X, window, axis=1)
    out[:, half : n_ch - half] = windows @ c

    t = np.arange(window, dtype=float)
    fit = np.linalg.pinv(np.vander(t, polyorder + 1, increasing=True))  # (p+1, window)
    head = _poly_deriv_matrix(t[:half], polyorder, deriv)
    tail = _poly_deriv_matrix(t[window - half :], polyorder, deriv)
    out[:, :half] = (X[:, :window] @ fit.T) @ head.T / delta**deriv
    out[:, n_ch - half :] = (X[:, -window:] @ fit.T) @ tail.T / delta**deriv
    return out


class SavitzkyGolay(_Stateless):
    def __init__(self, window: int = 11, polyorder: int = 2, deriv: int = 0, delta: float = 1.0):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def transform(self, X):
        return savgol(X, self.window, self.polyorder, self.deriv, self.delta)


# --------------------------------------------------------------------------- #
# Scatter correction
# --------------------------------------------------------------------------- #


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: centre and scale each spectrum by its own mean/std."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


class SNV(_Stateless):
    def transform(self, X):
        return snv(X)


class MSC(TransformerMixin, BaseEstimator):
    """Multiplicative Scatter Correction against the training-set mean spectrum."""

    def fit(self, X, y=None):
        self.reference_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        ref = self.reference_ - self.reference_.mean()
        Xc = X - X.mean(axis=1, keepdims=True)
        slope = Xc @ ref / (ref @ ref)
        offset = X.mean(axis=1) - slope * self.reference_.mean()
        return (X - offset[:, None]) / slope[:, None]


# --------------------------------------------------------------------------- #
# Baseline correction
# --------------------------------------------------------------------------- #


def poly_detrend(X: np.ndarray, order: int = 2) -> np.ndarray:
    """Subtract a least-squares polynomial (over channel index) from each spectrum."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    t = np.linspace(-1.0, 1.0, X.shape[1])
    V = np.vander(t, order + 1, increasing=True)
    coef = X @ np.linalg.pinv(V).T
    return X - coef @ V.T


def als_baseline(x: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens, 2005) for one spectrum."""
    L = x.size
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam * (D @ D.T)
    w = np.ones(L)
    z = x
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + H).tocsc(), w * x)
        w = p * (x > z) + (1 - p) * (x <= z)
    return z


class Baseline(_Stateless):
    """Baseline removal: ``method='poly'`` (detrend) or ``'als'`` (asymmetric least squares)."""

    def __init__(self, method: str = "poly", order: int = 2, lam: float = 1e5, p: float = 0.01):
        self.method = method
        self.order = order
        self.lam = lam
        self.p = p

    def fit(self, X, y=None):
        if self.method not in ("poly", "als"):
            raise ValueError(f"unknown baseline method {self.method!r}")
        return self

    def transform(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.method == "poly":
            return poly_detrend(X, self.order)
        return np.vstack([x - als_baseline(x, self.lam, self.p) for x in X])


# --------------------------------------------------------------------------- #
# Pipeline configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreprocessConfig:
    """Declarative description of a preprocessing chain: baseline -> scatter -> SG."""

    baseline: str | None = None  # None | "poly" | "als"
    scatter: str | None = None  # None | "snv" | "msc"
    sg_deriv: int | None = None  # None (no SG) | 0 (smooth) | 1 | 2
    sg_window: int = 15
    sg_polyorder: int = 2

    @property
    def name(self) -> str:
        parts = []
        if self.baseline:
            parts.append(f"base:{self.baseline}")
        if self.scatter:
            parts.append(self.scatter)
        if self.sg_deriv is not None:
            parts.append("sg-smooth" if self.sg_deriv == 0 else f"sg-d{self.sg_deriv}")
        return "+".join(parts) or "raw"

    def build(self) -> Pipeline:
        steps = []
        if self.baseline:
            steps.append(("baseline", Baseline(method=self.baseline)))
        if self.scatter == "snv":
            steps.append(("scatter", SNV()))
        elif self.scatter == "msc":
            steps.append(("scatter", MSC()))
        elif self.scatter is not None:
            raise ValueError(f"unknown scatter correction {self.scatter!r}")
        if self.sg_deriv is not None:
            polyorder = max(self.sg_polyorder, self.sg_deriv)
            steps.append(("sg", SavitzkyGolay(self.sg_window, polyorder, self.sg_deriv)))
        if not steps:
            steps.append(("identity", FunctionTransformer().fit(np.zeros((1, 1)))))
        return Pipeline(steps)


DEFAULT_PREPROCESS = PreprocessConfig(baseline="poly", scatter="snv", sg_deriv=1)

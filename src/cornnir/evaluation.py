"""Metrics, sample selection and cross-validation helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from .preprocess import PreprocessConfig


def rmse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Column-wise root mean squared error."""
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2, axis=0))


def regression_report(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """RMSE, bias, bias-corrected SEP, R^2 and RPD (SD / RMSE), column-wise."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    resid = pred - y
    bias = resid.mean(axis=0)
    sep = resid.std(axis=0, ddof=1)
    r = rmse(y, pred)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - (resid**2).sum(axis=0) / ss_tot
    rpd = y.std(axis=0, ddof=1) / r
    return {"rmse": r, "bias": bias, "sep": sep, "r2": r2, "rpd": rpd}


def kennard_stone(X: np.ndarray, n: int) -> np.ndarray:
    """Kennard-Stone selection: indices of ``n`` samples that uniformly span X-space."""
    X = np.asarray(X, float)
    if not 2 <= n <= len(X):
        raise ValueError("n must be between 2 and n_samples")
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    selected = [int(i), int(j)]
    min_d = np.minimum(d[i], d[j])
    min_d[selected] = -1
    while len(selected) < n:
        k = int(np.argmax(min_d))
        selected.append(k)
        min_d = np.minimum(min_d, d[k])
        min_d[selected] = -1
    return np.array(selected)


def train_test_split_ks(X: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Kennard-Stone calibration set plus the remaining samples as test set."""
    train = kennard_stone(X, n_train)
    test = np.setdiff1d(np.arange(len(X)), train)
    return np.sort(train), test


ModelFactory = Callable[[], RegressorMixin]


def make_pipeline(config: PreprocessConfig, model: RegressorMixin) -> Pipeline:
    return Pipeline([("pre", config.build()), ("model", model)])


def cross_validate(
    config: PreprocessConfig,
    model_factory: ModelFactory,
    X: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """Repeated K-fold RMSECV. Returns array of shape (n_repeats, n_targets)."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    out = np.zeros((n_repeats, Y.shape[1]))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    pred = np.zeros_like(Y)
    for f, (tr, te) in enumerate(rkf.split(X)):
        pipe = make_pipeline(config, model_factory()).fit(X[tr], Y[tr])
        pred[te] = np.asarray(pipe.predict(X[te])).reshape(len(te), -1)
        if (f + 1) % n_splits == 0:
            out[f // n_splits] = rmse(Y, pred)
    return out

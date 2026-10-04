"""PLS1 regression (NIPALS) in NumPy with cross-validated choice of latent variables."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold


def nipals_pls1(X: np.ndarray, y: np.ndarray, n_components: int):
    """Fit PLS1 on centred data. Returns (W, P, q) with W, P of shape (n_ch, A)."""
    Xr = X.copy()
    yr = y.copy()
    n_ch = X.shape[1]
    W = np.zeros((n_ch, n_components))
    P = np.zeros((n_ch, n_components))
    q = np.zeros(n_components)
    for a in range(n_components):
        w = Xr.T @ yr
        norm = np.linalg.norm(w)
        if norm < 1e-12:  # y fully explained; remaining components are zero
            W, P, q = W[:, :a], P[:, :a], q[:a]
            break
        w /= norm
        t = Xr @ w
        tt = t @ t
        p = Xr.T @ t / tt
        q[a] = yr @ t / tt
        Xr -= np.outer(t, p)
        yr = yr - q[a] * t
        W[:, a], P[:, a] = w, p
    return W, P, q


class PLSRegressor(BaseEstimator, RegressorMixin):
    """Single-response PLS regression with mean-centring (no variance scaling).

    ``coefs_[k-1]`` holds the regression vector using ``k`` latent variables, so
    predictions for every model size come from one fit (used for CV selection).
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        A = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        W, P, q = nipals_pls1(X - self.x_mean_, y - self.y_mean_, A)
        A = W.shape[1]
        self.coefs_ = np.zeros((A, X.shape[1]))
        for k in range(1, A + 1):
            R = W[:, :k] @ np.linalg.inv(P[:, :k].T @ W[:, :k])
            self.coefs_[k - 1] = R @ q[:k]
        self.W_, self.P_, self.q_ = W, P, q
        self.n_components_ = A
        return self

    def predict_all(self, X) -> np.ndarray:
        """Predictions for 1..A components, shape (n_samples, A)."""
        Xc = np.asarray(X, dtype=float) - self.x_mean_
        return Xc @ self.coefs_.T + self.y_mean_

    def predict(self, X, n_components: int | None = None) -> np.ndarray:
        k = n_components or self.n_components_
        Xc = np.asarray(X, dtype=float) - self.x_mean_
        return Xc @ self.coefs_[k - 1] + self.y_mean_


def select_components(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 20,
    cv: int = 5,
    tol: float = 0.02,
    random_state: int = 0,
) -> tuple[int, np.ndarray]:
    """Return (k, rmsecv_curve): the smallest k whose RMSECV is within ``tol`` of the minimum."""
    max_components = min(max_components, int(len(y) * (cv - 1) / cv) - 1, X.shape[1])
    press = np.zeros(max_components)
    folds = KFold(cv, shuffle=True, random_state=random_state)
    for tr, te in folds.split(X):
        pls = PLSRegressor(max_components).fit(X[tr], y[tr])
        pred = pls.predict_all(X[te])
        a = pred.shape[1]
        press[:a] += ((pred - y[te, None]) ** 2).sum(axis=0)
        press[a:] = np.inf  # a fold could not support that many components
    rmsecv = np.sqrt(press / len(y))
    k = int(np.argmax(rmsecv <= rmsecv.min() * (1 + tol))) + 1
    return k, rmsecv


class PLSModel(BaseEstimator, RegressorMixin):
    """One PLS1 per target column, each with its own CV-selected number of components."""

    def __init__(
        self,
        max_components: int = 20,
        cv: int = 5,
        n_components: int | None = None,
        random_state: int = 0,
    ):
        self.max_components = max_components
        self.cv = cv
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self._1d = Y.ndim == 1
        Y = Y.reshape(len(Y), -1)
        self.models_, self.rmsecv_ = [], []
        for j in range(Y.shape[1]):
            if self.n_components is None:
                k, curve = select_components(
                    X, Y[:, j], self.max_components, self.cv, random_state=self.random_state
                )
            else:
                k, curve = self.n_components, None
            self.models_.append(PLSRegressor(k).fit(X, Y[:, j]))
            self.rmsecv_.append(curve)
        self.n_components_ = [m.n_components_ for m in self.models_]
        return self

    def predict(self, X):
        pred = np.column_stack([m.predict(X) for m in self.models_])
        return pred.ravel() if self._1d else pred

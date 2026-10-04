"""Drift detection for spectra: PCA-based T^2 / Q monitoring plus an EWMA chart.

The detector learns the normal spectral subspace from reference (calibration)
spectra. New spectra are flagged when their Hotelling T^2 (unusual position
inside the subspace) or Q / SPE (variation outside it) exceeds a control limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from .preprocess import PreprocessConfig


class _PCA:
    def __init__(self, n_components: int | None, var_explained: float):
        self.n_components = n_components
        self.var_explained = var_explained

    def fit(self, X: np.ndarray) -> _PCA:
        self.mean_ = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        eig = s**2 / (len(X) - 1)
        if self.n_components is None:
            k = int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.var_explained) + 1)
        else:
            k = self.n_components
        self.k_ = min(k, len(X) - 2)
        self.P_ = Vt[: self.k_].T
        self.eig_ = eig[: self.k_]
        self.resid_eig_ = eig[self.k_ :]
        return self

    def stats(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = X - self.mean_
        T = Xc @ self.P_
        t2 = ((T**2) / self.eig_).sum(1)
        q = ((Xc - T @ self.P_.T) ** 2).sum(1)
        return t2, q


@dataclass
class DriftReport:
    n: int
    n_flagged: int
    expected_rate: float
    p_value: float
    drift: bool
    t2: np.ndarray
    q: np.ndarray
    flags: np.ndarray

    @property
    def flagged_fraction(self) -> float:
        return self.n_flagged / self.n


class PCADriftDetector:
    """Hotelling T^2 / Q-residual detector with a batch-level binomial test.

    ``limits='cv'`` sets control limits as the (1-alpha) quantile of statistics
    computed on *held-out* reference spectra via K-fold, which is honest about the
    fact that training samples always fit their own PCA model better than new ones.
    ``limits='parametric'`` uses the F-distribution (T^2) and Jackson-Mudholkar (Q).
    """

    def __init__(
        self,
        preprocess: PreprocessConfig | None = None,
        n_components: int | None = None,
        var_explained: float = 0.99,
        alpha: float = 0.01,
        limits: str = "cv",
        cv: int = 5,
        random_state: int = 0,
    ):
        self.preprocess = preprocess
        self.n_components = n_components
        self.var_explained = var_explained
        self.alpha = alpha
        self.limits = limits
        self.cv = cv
        self.random_state = random_state

    def _prep(self, X):
        X = np.asarray(X, float)
        return self.pre_.transform(X) if self.pre_ is not None else X

    def fit(self, X_ref: np.ndarray):
        self.pre_ = self.preprocess.build().fit(X_ref) if self.preprocess else None
        Z = self._prep(X_ref)
        self.pca_ = _PCA(self.n_components, self.var_explained).fit(Z)
        k, n = self.pca_.k_, len(Z)
        if self.limits == "parametric":
            self.t2_limit_ = (
                k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(1 - self.alpha, k, n - k)
            )
            self.q_limit_ = _jackson_mudholkar(self.pca_.resid_eig_, self.alpha)
        elif self.limits == "cv":
            t2s, qs = [], []
            for tr, te in KFold(self.cv, shuffle=True, random_state=self.random_state).split(Z):
                pca = _PCA(k, self.var_explained).fit(Z[tr])
                t2, q = pca.stats(Z[te])
                t2s.append(t2)
                qs.append(q)
            self.cv_t2_, self.cv_q_ = np.concatenate(t2s), np.concatenate(qs)
            self.t2_limit_ = float(np.quantile(self.cv_t2_, 1 - self.alpha))
            self.q_limit_ = float(np.quantile(self.cv_q_, 1 - self.alpha))
        else:
            raise ValueError(f"unknown limits {self.limits!r}")
        # a sample is flagged if either statistic exceeds its limit
        self.expected_rate_ = 1 - (1 - self.alpha) ** 2
        return self

    def statistics(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.pca_.stats(self._prep(X))

    def flags(self, X: np.ndarray) -> np.ndarray:
        t2, q = self.statistics(X)
        return (t2 > self.t2_limit_) | (q > self.q_limit_)

    def test_batch(self, X: np.ndarray, significance: float = 0.01) -> DriftReport:
        t2, q = self.statistics(X)
        flags = (t2 > self.t2_limit_) | (q > self.q_limit_)
        k = int(flags.sum())
        p = float(stats.binomtest(k, len(flags), self.expected_rate_, alternative="greater").pvalue)
        return DriftReport(len(flags), k, self.expected_rate_, p, p < significance, t2, q, flags)


def _jackson_mudholkar(resid_eig: np.ndarray, alpha: float) -> float:
    th1, th2, th3 = (np.sum(resid_eig**i) for i in (1, 2, 3))
    h0 = 1 - 2 * th1 * th3 / (3 * th2**2)
    z = stats.norm.ppf(1 - alpha)
    return float(
        th1 * (z * np.sqrt(2 * th2 * h0**2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1**2) ** (1 / h0)
    )


class EWMAMonitor:
    """Exponentially weighted moving average chart for a stream of scalar statistics.

    Fitted on in-control values (e.g. log Q of held-out reference spectra); signals
    when the EWMA exceeds mean + L * sigma * sqrt(lambda / (2 - lambda)).
    """

    def __init__(self, lam: float = 0.2, L: float = 3.0):
        self.lam = lam
        self.L = L

    def fit(self, in_control: np.ndarray):
        self.mu_ = float(np.mean(in_control))
        self.sigma_ = float(np.std(in_control, ddof=1))
        self.limit_ = self.mu_ + self.L * self.sigma_ * np.sqrt(self.lam / (2 - self.lam))
        self.reset()
        return self

    def reset(self) -> None:
        self.z_ = self.mu_

    def update(self, value: float) -> tuple[float, bool]:
        self.z_ = self.lam * value + (1 - self.lam) * self.z_
        return self.z_, self.z_ > self.limit_

    def run(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.reset()
        out = [self.update(v) for v in values]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

import numpy as np
import pytest
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from cornnir.models.pls import PLSRegressor
from cornnir.preprocess import (
    MSC,
    SNV,
    Baseline,
    PreprocessConfig,
    SavitzkyGolay,
    als_baseline,
    poly_detrend,
    savgol,
    savgol_coeffs,
    snv,
)


@pytest.mark.parametrize("window,poly,deriv", [(5, 2, 0), (11, 2, 1), (15, 3, 2), (21, 4, 1)])
def test_savgol_matches_scipy(spectra, window, poly, deriv):
    X, _ = spectra
    ours = savgol(X, window, poly, deriv, delta=2.0)
    ref = savgol_filter(X, window, poly, deriv=deriv, delta=2.0, axis=1, mode="interp")
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_savgol_preserves_polynomials():
    t = np.arange(50, dtype=float)
    x = 3 + 0.5 * t - 0.01 * t**2
    np.testing.assert_allclose(savgol(x, 7, 2)[0], x, atol=1e-9)
    np.testing.assert_allclose(savgol(x, 7, 2, deriv=1)[0], 0.5 - 0.02 * t, atol=1e-9)


def test_savgol_coeffs_validation():
    with pytest.raises(ValueError):
        savgol_coeffs(4, 2)
    with pytest.raises(ValueError):
        savgol_coeffs(5, 5)
    with pytest.raises(ValueError):
        savgol(np.ones((2, 5)), 7, 2)


def test_snv_row_stats(spectra):
    Z = snv(spectra[0])
    np.testing.assert_allclose(Z.mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(1, ddof=1), 1, atol=1e-12)


def test_snv_removes_affine_scatter(spectra):
    x = spectra[0][:1]
    np.testing.assert_allclose(snv(2.5 * x + 0.3), snv(x), atol=1e-12)


def test_snv_constant_row_does_not_divide_by_zero():
    assert np.all(np.isfinite(snv(np.ones((1, 10)))))


def test_msc_removes_scatter_relative_to_reference(spectra):
    X, _ = spectra
    msc = MSC().fit(X)
    distorted = 1.3 * msc.reference_ + 0.1
    np.testing.assert_allclose(msc.transform(distorted[None])[0], msc.reference_, atol=1e-10)


def test_poly_detrend_removes_polynomial(spectra):
    X = spectra[0]
    t = np.linspace(-1, 1, X.shape[1])
    trend = 0.2 + 0.1 * t + 0.3 * t**2
    np.testing.assert_allclose(poly_detrend(X + trend, 2), poly_detrend(X, 2), atol=1e-10)


def test_als_baseline_tracks_baseline_under_peak():
    t = np.linspace(0, 1, 300)
    baseline = 0.5 + 0.3 * t
    x = baseline + np.exp(-((t - 0.5) ** 2) / (2 * 0.02**2))
    z = als_baseline(x, lam=1e5, p=0.001)
    assert np.max(np.abs(z - baseline)) < 0.05


def test_baseline_rejects_unknown_method():
    with pytest.raises(ValueError):
        Baseline(method="nope").fit(np.ones((2, 5)))


def test_config_names_and_build():
    assert PreprocessConfig().name == "raw"
    cfg = PreprocessConfig(baseline="poly", scatter="snv", sg_deriv=1)
    assert cfg.name == "base:poly+snv+sg-d1"
    steps = [name for name, _ in cfg.build().steps]
    assert steps == ["baseline", "scatter", "sg"]
    assert PreprocessConfig(sg_deriv=0).name == "sg-smooth"
    with pytest.raises(ValueError):
        PreprocessConfig(scatter="bogus").build()


@pytest.mark.parametrize(
    "cfg", [PreprocessConfig(), PreprocessConfig(baseline="poly", scatter="msc", sg_deriv=2)]
)
def test_pipelines_shape_and_sklearn_compat(spectra, cfg):
    X, C = spectra
    pipe = cfg.build()
    assert pipe.fit_transform(X).shape == X.shape
    model = Pipeline([("pre", cfg.build()), ("pls", PLSRegressor(3))])
    scores = cross_val_score(clone(model), X, C[:, 0], cv=3)
    assert np.all(np.isfinite(scores))


def test_transformers_are_stateless_where_expected(spectra):
    X = spectra[0]
    for tr in (SNV(), SavitzkyGolay(), Baseline()):
        np.testing.assert_allclose(tr.transform(X), clone(tr).fit(X[:5]).transform(X))

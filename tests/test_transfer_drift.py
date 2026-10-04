import numpy as np
import pytest

from cornnir.drift import EWMAMonitor, PCADriftDetector
from cornnir.evaluation import kennard_stone, regression_report, rmse
from cornnir.preprocess import PreprocessConfig
from cornnir.transfer import (
    DirectStandardization,
    PiecewiseDirectStandardization,
    SlopeBiasCorrection,
)

from .conftest import make_spectra


def simulate_slave(X, shift=2, gain=1.05, offset=0.02):
    """A second instrument: wavelength shift by `shift` channels, gain, offset, slight blur."""
    Xs = np.roll(X, shift, axis=1)
    Xs[:, :shift] = Xs[:, [shift]]
    kernel = np.array([0.25, 0.5, 0.25])
    Xs = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, Xs)
    return gain * Xs + offset


@pytest.fixture
def pair():
    X, C = make_spectra(n=80, seed=3)
    return X, simulate_slave(X), C


def test_pds_recovers_master_spectra(pair):
    Xm, Xs, _ = pair
    std = kennard_stone(Xm[:60], 12)
    pds = PiecewiseDirectStandardization(half_window=4, rcond=1e-3).fit(Xm[std], Xs[std])
    before = np.abs(Xs[60:] - Xm[60:]).mean()
    after = np.abs(pds.transform(Xs[60:]) - Xm[60:]).mean()
    assert after < 0.2 * before
    # transfer matrix is banded
    F = pds.F_
    i, j = np.nonzero(F)
    assert np.max(np.abs(i - j)) <= 4


def test_ds_exact_when_enough_standards():
    rng = np.random.default_rng(0)
    Xm = rng.normal(size=(40, 10))
    A = np.eye(10) + 0.1 * rng.normal(size=(10, 10))
    Xs = Xm @ np.linalg.inv(A) + 0.5
    ds = DirectStandardization(rcond=1e-12).fit(Xm[:20], Xs[:20])
    np.testing.assert_allclose(ds.transform(Xs[20:]), Xm[20:], atol=1e-8)


def test_pds_shape_validation():
    with pytest.raises(ValueError):
        PiecewiseDirectStandardization().fit(np.ones((3, 10)), np.ones((3, 9)))


def test_slope_bias_correction():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(30, 2))
    pred = 0.8 * y + np.array([1.0, -2.0])
    sbc = SlopeBiasCorrection().fit(pred[:10], y[:10])
    np.testing.assert_allclose(sbc.transform(pred[10:]), y[10:], atol=1e-10)


@pytest.mark.parametrize("limits", ["cv", "parametric"])
def test_detector_flags_shifted_instrument(pair, limits):
    Xm, Xs, _ = pair
    det = PCADriftDetector(limits=limits, alpha=0.01).fit(Xm[:60])
    in_control = det.test_batch(Xm[60:])
    shifted = det.test_batch(Xs[60:])
    assert in_control.flagged_fraction <= 0.15 and not in_control.drift
    assert shifted.flagged_fraction > 0.9 and shifted.drift
    assert shifted.p_value < 1e-6


def test_detector_with_preprocessing(pair):
    Xm, Xs, _ = pair
    det = PCADriftDetector(preprocess=PreprocessConfig(scatter="snv", sg_deriv=1)).fit(Xm[:60])
    assert det.test_batch(Xs[60:]).drift


def test_detector_rejects_bad_limits(pair):
    with pytest.raises(ValueError):
        PCADriftDetector(limits="magic").fit(pair[0])


def test_ewma_alarms_only_after_shift():
    rng = np.random.default_rng(0)
    mon = EWMAMonitor(lam=0.2, L=3).fit(rng.normal(size=500))
    _, alarms_ok = mon.run(rng.normal(size=200))
    assert alarms_ok.mean() < 0.02
    stream = np.r_[rng.normal(size=50), rng.normal(loc=2, size=50)]
    _, alarms = mon.run(stream)
    first = int(np.argmax(alarms))
    assert alarms.any() and first >= 50 and first < 60


def test_kennard_stone():
    X = np.array([[0.0], [10.0], [5.0], [1.0], [9.0]])
    sel = kennard_stone(X, 3)
    assert set(sel[:2]) == {0, 1} and sel[2] == 2
    assert len(set(kennard_stone(np.random.default_rng(0).normal(size=(30, 4)), 30))) == 30
    with pytest.raises(ValueError):
        kennard_stone(X, 1)


def test_regression_report():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    rep = regression_report(y, y + 0.5)
    np.testing.assert_allclose(rep["bias"], 0.5)
    np.testing.assert_allclose(rep["sep"], 0.0, atol=1e-12)
    np.testing.assert_allclose(rep["rmse"], rmse(y, y + 0.5))

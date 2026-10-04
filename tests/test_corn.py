"""Integration tests on the real Eigenvector corn data (downloaded on first use)."""

import numpy as np
import pytest

from cornnir import experiments as ex
from cornnir.evaluation import cross_validate
from cornnir.preprocess import PreprocessConfig

pytestmark = pytest.mark.data


def test_dataset_shapes(corn):
    assert corn.wavelengths[0] == 1100 and corn.wavelengths[-1] == 2498
    for inst in ("m5", "mp5", "mp6"):
        assert corn.X(inst).shape == (80, 700)
    assert corn.y.shape == (80, 4)
    assert corn.target("starch").mean() > 60
    with pytest.raises(KeyError):
        corn.X("nope")


def test_split_is_deterministic(corn):
    a, b = ex.standard_split(corn), ex.standard_split(corn)
    np.testing.assert_array_equal(a.train, b.train)
    assert len(a.train) == 60 and len(a.test) == 20
    assert not set(a.train) & set(a.test)


def test_pls_cv_is_reasonable(corn):
    scores = cross_validate(PreprocessConfig(), ex.pls_factory, corn.X("m5"), corn.y, n_repeats=1)
    rel = scores[0] / corn.y.std(0, ddof=1)
    assert np.all(rel < 0.5)  # every property clearly better than predicting the mean


def test_instrument_shift_collapses_and_transfer_recovers(corn):
    err, flags = ex.run_transfer(corn, n_standards=(10,))
    mo = err[(err.slave == "mp5") & (err.property == "moisture")].set_index("method").rmse
    assert mo["none"] > 50 * mo["master (reference)"]  # collapse
    assert mo["PDS"] < mo["none"] / 3  # recovery
    f = flags.set_index(["slave", "method"]).flagged
    assert f[("mp5", "none")] == 1.0 and f[("m5", "none")] <= 0.1
    assert f[("mp5", "PDS")] <= 0.1


def test_streaming_recalibration_helps(corn):
    stream, recal = ex.run_streaming(corn, n_steps=200, drift_start=40, drift_end=140)
    assert recal and recal[0] > 40
    after = stream[stream.t >= 140]
    static = np.sqrt((after.err_static_moisture**2).mean())
    monitored = np.sqrt((after.err_monitored_moisture**2).mean())
    assert monitored < static / 3
    assert not stream[stream.t < 40].alarm.any()

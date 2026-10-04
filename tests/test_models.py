import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from cornnir.models.cnn import CNNRegressor
from cornnir.models.pls import PLSModel, PLSRegressor, select_components


@pytest.mark.parametrize("k", [1, 3, 6])
def test_pls_matches_sklearn(spectra, k):
    X, C = spectra
    y = C[:, 1]
    ours = PLSRegressor(k).fit(X, y).predict(X)
    ref = PLSRegression(k, scale=False).fit(X, y).predict(X).ravel()
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_predict_all_consistent(spectra):
    X, C = spectra
    pls = PLSRegressor(5).fit(X, C[:, 0])
    allp = pls.predict_all(X)
    assert allp.shape == (len(X), 5)
    for k in range(1, 6):
        np.testing.assert_allclose(allp[:, k - 1], pls.predict(X, k))


def test_pls_caps_components_at_rank():
    X = np.random.default_rng(0).normal(size=(6, 50))
    pls = PLSRegressor(20).fit(X, X[:, 0])
    assert pls.n_components_ <= 5


def test_select_components_finds_low_rank(spectra):
    X, C = spectra
    k, curve = select_components(X, C[:, 0], max_components=12)
    # 3 bands + multiplicative/additive scatter -> a handful of LVs suffice
    assert 2 <= k <= 8
    assert curve.shape == (12,)
    assert curve[k - 1] <= curve.min() * 1.02


def test_plsmodel_multioutput(spectra):
    X, C = spectra
    m = PLSModel(max_components=10).fit(X, C)
    assert m.predict(X).shape == C.shape
    assert len(m.n_components_) == C.shape[1]
    assert PLSModel(n_components=2).fit(X, C[:, 0]).predict(X).shape == (len(X),)


def test_cnn_learns_and_is_deterministic(spectra):
    X, C = spectra
    kw = dict(epochs=150, patience=150, random_state=1)
    a = CNNRegressor(**kw).fit(X, C)
    pred = a.predict(X)
    assert pred.shape == C.shape
    mse = ((pred - C) ** 2).mean()
    assert mse < 0.5 * C.var(0).mean()
    np.testing.assert_allclose(pred, CNNRegressor(**kw).fit(X, C).predict(X), rtol=1e-5)


def test_cnn_1d_target_and_ensemble(spectra):
    X, C = spectra
    m = CNNRegressor(epochs=5, n_ensemble=2, augment_strength=1.0, val_fraction=0).fit(X, C[:, 0])
    assert m.predict(X).shape == (len(X),)
    assert len(m.nets_) == 2

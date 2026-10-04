import numpy as np
import pytest


def make_spectra(n=60, n_ch=200, n_comp=3, noise=1e-3, seed=0):
    """Synthetic absorbance spectra: Gaussian bands mixed by concentrations + scatter."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_ch)
    centres = np.linspace(0.2, 0.8, n_comp)
    bands = np.stack([np.exp(-((t - c) ** 2) / (2 * 0.04**2)) for c in centres])
    C = rng.uniform(0.5, 1.5, size=(n, n_comp))
    X = C @ bands
    scale = rng.uniform(0.9, 1.1, size=(n, 1))
    offset = rng.uniform(-0.05, 0.05, size=(n, 1))
    X = X * scale + offset + noise * rng.standard_normal(X.shape)
    return X, C


@pytest.fixture
def spectra():
    return make_spectra()


@pytest.fixture(scope="session")
def corn():
    from cornnir.data import load_corn

    try:
        return load_corn()
    except Exception as exc:  # pragma: no cover - offline machines
        pytest.skip(f"corn dataset unavailable: {exc}")

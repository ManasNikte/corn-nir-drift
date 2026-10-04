"""Download, verify and load the Eigenvector Research corn NIR dataset.

80 corn samples measured on three NIR spectrometers (m5, mp5, mp6),
1100-2498 nm at 2 nm (700 channels), with moisture/oil/protein/starch
reference values and NBS glass standards measured on each instrument.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

URL = "https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip"
ZIP_SHA256 = "8a2d1a03648b6ad334caaafa5d8377bf945ba1477a5082c4963d705d07cca795"
INSTRUMENTS = ("m5", "mp5", "mp6")
PROPERTIES = ("moisture", "oil", "protein", "starch")


def default_data_dir() -> Path:
    return Path(os.environ.get("CORNNIR_DATA_DIR", Path.home() / ".cache" / "cornnir"))


def download(data_dir: Path | None = None, force: bool = False) -> Path:
    """Fetch the zip (if needed), verify its checksum and return the path to corn.mat."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "corn.zip"
    mat_path = data_dir / "corn.mat"
    if mat_path.exists() and not force:
        return mat_path

    if force or not zip_path.exists():
        with urllib.request.urlopen(URL, timeout=60) as resp:  # noqa: S310 - fixed https URL
            payload = resp.read()
        zip_path.write_bytes(payload)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        zip_path.unlink()
        raise RuntimeError(f"corn.zip checksum mismatch: got {digest}, expected {ZIP_SHA256}")

    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        mat_path.write_bytes(zf.read("corn.mat"))
    return mat_path


@dataclass(frozen=True)
class CornDataset:
    wavelengths: np.ndarray  # (700,) nm
    spectra: dict[str, np.ndarray]  # instrument -> (80, 700) absorbance
    y: np.ndarray  # (80, 4)
    properties: tuple[str, ...] = PROPERTIES
    glass_standards: dict[str, np.ndarray] = field(default_factory=dict)

    def X(self, instrument: str) -> np.ndarray:
        if instrument not in self.spectra:
            raise KeyError(f"unknown instrument {instrument!r}; choose from {list(self.spectra)}")
        return self.spectra[instrument]

    def target(self, prop: str) -> np.ndarray:
        return self.y[:, self.properties.index(prop)]

    @property
    def n_samples(self) -> int:
        return self.y.shape[0]


def _dataset_field(mat: dict, key: str, name: str) -> np.ndarray:
    return np.asarray(mat[key][0, 0][name])


def load_corn(data_dir: Path | None = None) -> CornDataset:
    mat = sio.loadmat(download(data_dir))
    spectra = {
        inst: _dataset_field(mat, f"{inst}spec", "data").astype(float) for inst in INSTRUMENTS
    }
    standards = {
        inst: _dataset_field(mat, f"{inst}nbs", "data").astype(float) for inst in INSTRUMENTS
    }
    axis = _dataset_field(mat, "m5spec", "axisscale")
    wavelengths = np.asarray(axis[1, 0], dtype=float).ravel()
    y = _dataset_field(mat, "propvals", "data").astype(float)
    return CornDataset(wavelengths=wavelengths, spectra=spectra, y=y, glass_standards=standards)

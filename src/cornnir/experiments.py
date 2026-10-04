"""End-to-end experiments. Each returns tidy DataFrames; ``cli.py`` writes them to disk."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import INSTRUMENTS, CornDataset
from .drift import EWMAMonitor, PCADriftDetector
from .evaluation import cross_validate, kennard_stone, make_pipeline, rmse, train_test_split_ks
from .models.cnn import CNNRegressor
from .models.pls import PLSModel
from .preprocess import DEFAULT_PREPROCESS, PreprocessConfig
from .transfer import DirectStandardization, PiecewiseDirectStandardization, SlopeBiasCorrection

N_TRAIN = 60
MASTER = "m5"


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray


def standard_split(ds: CornDataset) -> Split:
    """Kennard-Stone on the master spectra: 60 calibration samples, 20 test samples."""
    return Split(*train_test_split_ks(ds.X(MASTER), N_TRAIN))


def pls_factory():
    return PLSModel(max_components=25)


def cnn_factory(**kw):
    def make():
        return CNNRegressor(**kw)

    return make


def _tidy(values: np.ndarray, ds: CornDataset, **cols) -> list[dict]:
    sd = ds.y.std(axis=0, ddof=1)
    return [
        {**cols, "property": p, "rmse": float(v), "rel_rmse": float(v / sd[j])}
        for j, (p, v) in enumerate(zip(ds.properties, values, strict=True))
    ]


# --------------------------------------------------------------------------- #
# 1-2. Preprocessing ablation and PLS vs CNN
# --------------------------------------------------------------------------- #


def ablation_grid() -> list[PreprocessConfig]:
    """Full factorial: baseline {off, poly} x scatter {off, SNV} x SG {off, smooth, d1, d2}."""
    return [
        PreprocessConfig(baseline=b, scatter=s, sg_deriv=g)
        for b, s, g in itertools.product([None, "poly"], [None, "snv"], [None, 0, 1, 2])
    ]


def _cv_job(config, factory, X, Y, n_repeats, seed):
    import torch

    torch.set_num_threads(1)
    return config, cross_validate(config, factory, X, Y, n_repeats=n_repeats, random_state=seed)


def run_ablation(
    ds: CornDataset,
    include_cnn: bool = True,
    n_jobs: int = -1,
    pls_repeats: int = 3,
    cnn_repeats: int = 1,
) -> pd.DataFrame:
    """RMSECV on the master instrument (all 80 samples) for every preprocessing config."""
    X, Y = ds.X(MASTER), ds.y
    configs = ablation_grid()
    extras = [
        PreprocessConfig(baseline="als"),
        PreprocessConfig(scatter="msc"),
        PreprocessConfig(scatter="msc", sg_deriv=1),
    ]
    jobs = [("PLS", c, pls_factory, pls_repeats) for c in configs + extras]
    if include_cnn:
        jobs += [("CNN", c, cnn_factory(), cnn_repeats) for c in configs]

    results = Parallel(n_jobs=n_jobs)(delayed(_cv_job)(c, f, X, Y, r, 0) for _, c, f, r in jobs)
    rows = []
    for (model, *_), (config, scores) in zip(jobs, results, strict=True):
        for rep, vals in enumerate(scores):
            rows += _tidy(
                vals,
                ds,
                model=model,
                config=config.name,
                baseline=config.baseline or "-",
                scatter=config.scatter or "-",
                sg="-"
                if config.sg_deriv is None
                else ("smooth" if config.sg_deriv == 0 else f"d{config.sg_deriv}"),
                factorial=config in configs,
                repeat=rep,
            )
    return pd.DataFrame(rows)


def main_effects(ablation: pd.DataFrame) -> pd.DataFrame:
    """Average change in relative RMSECV when each factor level is switched on vs off."""
    df = (
        ablation[ablation.factorial]
        .groupby(["model", "config", "baseline", "scatter", "sg"], as_index=False)
        .rel_rmse.mean()
    )
    rows = []
    for model, g in df.groupby("model"):
        for factor, off in [("baseline", "-"), ("scatter", "-"), ("sg", "-")]:
            base = g[g[factor] == off].rel_rmse.mean()
            for level in sorted(set(g[factor]) - {off}):
                rows.append(
                    {
                        "model": model,
                        "factor": factor,
                        "level": level,
                        "mean_rel_rmse_off": base,
                        "mean_rel_rmse_on": g[g[factor] == level].rel_rmse.mean(),
                    }
                )
    out = pd.DataFrame(rows)
    out["effect_pct"] = 100 * (out.mean_rel_rmse_on / out.mean_rel_rmse_off - 1)
    return out


def run_model_comparison(
    ds: CornDataset, configs: dict[str, PreprocessConfig] | None = None, n_jobs: int = -1
) -> pd.DataFrame:
    """Test-set RMSEP on the Kennard-Stone split for PLS and CNN variants."""
    split = standard_split(ds)
    X, Y = ds.X(MASTER), ds.y
    configs = configs or {
        "raw": PreprocessConfig(),
        "default": DEFAULT_PREPROCESS,
        "sg-d1": PreprocessConfig(sg_deriv=1),  # best PLS config in the ablation
    }
    variants = []
    for cname, cfg in configs.items():
        variants += [
            ("PLS", cname, cfg, pls_factory),
            ("CNN", cname, cfg, cnn_factory()),
            ("CNN+aug", cname, cfg, cnn_factory(augment_strength=1.0)),
            ("CNN+aug x5", cname, cfg, cnn_factory(augment_strength=1.0, n_ensemble=5)),
        ]

    def job(cfg, factory):
        import torch

        torch.set_num_threads(1)
        pipe = make_pipeline(cfg, factory()).fit(X[split.train], Y[split.train])
        return rmse(Y[split.test], pipe.predict(X[split.test]))

    scores = Parallel(n_jobs=n_jobs)(delayed(job)(cfg, f) for _, _, cfg, f in variants)
    rows = []
    for (model, cname, cfg, _), s in zip(variants, scores, strict=True):
        rows += _tidy(s, ds, model=model, preprocess=cname, config=cfg.name)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Instrument shift
# --------------------------------------------------------------------------- #


def run_instrument_shift(
    ds: CornDataset,
    configs: dict[str, PreprocessConfig] | None = None,
    include_cnn: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Train on one instrument's calibration samples, test on every instrument's test samples."""
    split = standard_split(ds)
    configs = configs or {"raw": PreprocessConfig(), "default": DEFAULT_PREPROCESS}
    models = [("PLS", pls_factory)] + ([("CNN", cnn_factory())] if include_cnn else [])
    jobs = [
        (m, f, cname, cfg, tr)
        for m, f in models
        for cname, cfg in configs.items()
        for tr in INSTRUMENTS
    ]

    def job(factory, cfg, train_inst):
        import torch

        torch.set_num_threads(1)
        pipe = make_pipeline(cfg, factory()).fit(ds.X(train_inst)[split.train], ds.y[split.train])
        return {
            te: rmse(ds.y[split.test], pipe.predict(ds.X(te)[split.test])) for te in INSTRUMENTS
        }

    results = Parallel(n_jobs=n_jobs)(delayed(job)(f, cfg, tr) for _, f, _, cfg, tr in jobs)
    rows = []
    for (m, _, cname, _, tr), res in zip(jobs, results, strict=True):
        for te, s in res.items():
            rows += _tidy(s, ds, model=m, preprocess=cname, train=tr, test=te)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Drift detection + calibration transfer
# --------------------------------------------------------------------------- #


def transfer_methods():
    return {
        "PDS": PiecewiseDirectStandardization(half_window=5, rcond=0.1),
        "DS": DirectStandardization(rcond=1e-3),
    }


def run_transfer(
    ds: CornDataset,
    n_standards=(3, 5, 8, 10, 15, 20, 30),
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Master m5 model applied to mp5 / mp6 test spectra, with and without transfer.

    Transfer standards are Kennard-Stone picks from the *calibration* samples, so the
    test samples are never used to fit the transfer. Returns (errors, drift flags).
    """
    split = standard_split(ds)
    cfg = preprocess or PreprocessConfig()
    Xm, Y = ds.X(MASTER), ds.y
    model = make_pipeline(cfg, pls_factory()).fit(Xm[split.train], Y[split.train])
    detector = PCADriftDetector(preprocess=cfg if cfg.name != "raw" else None).fit(Xm[split.train])
    yt = Y[split.test]

    err, flags = [], []
    master_rmse = rmse(yt, model.predict(Xm[split.test]))
    for slave in INSTRUMENTS[1:]:
        Xs = ds.X(slave)
        base = dict(slave=slave, preprocess=cfg.name)
        err += _tidy(master_rmse, ds, **base, method="master (reference)", n_standards=0)
        err += _tidy(
            rmse(yt, model.predict(Xs[split.test])), ds, **base, method="none", n_standards=0
        )
        flags.append(
            {
                **base,
                "method": "none",
                "n_standards": 0,
                "flagged": detector.test_batch(Xs[split.test]).flagged_fraction,
            }
        )
        for n in n_standards:
            std = split.train[kennard_stone(Xm[split.train], n)]
            sbc = SlopeBiasCorrection().fit(model.predict(Xs[std]), Y[std])
            err += _tidy(
                rmse(yt, sbc.transform(model.predict(Xs[split.test]))),
                ds,
                **base,
                method="SBC",
                n_standards=n,
            )
            for name, tf in transfer_methods().items():
                Xc = tf.fit(Xm[std], Xs[std]).transform(Xs[split.test])
                err += _tidy(rmse(yt, model.predict(Xc)), ds, **base, method=name, n_standards=n)
                flags.append(
                    {
                        **base,
                        "method": name,
                        "n_standards": n,
                        "flagged": detector.test_batch(Xc).flagged_fraction,
                    }
                )
    flags.append(
        {
            "slave": MASTER,
            "preprocess": cfg.name,
            "method": "none",
            "n_standards": 0,
            "flagged": detector.test_batch(Xm[split.test]).flagged_fraction,
        }
    )
    return pd.DataFrame(err), pd.DataFrame(flags)


def run_pds_grid(
    ds: CornDataset,
    slave: str = "mp5",
    n_std: int = 10,
    half_windows=(1, 2, 3, 5, 8, 12),
    rconds=(0.3, 0.1, 0.03, 0.01),
) -> pd.DataFrame:
    split = standard_split(ds)
    Xm, Xs, Y = ds.X(MASTER), ds.X(slave), ds.y
    model = make_pipeline(PreprocessConfig(), pls_factory()).fit(Xm[split.train], Y[split.train])
    std = split.train[kennard_stone(Xm[split.train], n_std)]
    rows = []
    for hw, rc in itertools.product(half_windows, rconds):
        pds = PiecewiseDirectStandardization(hw, rcond=rc).fit(Xm[std], Xs[std])
        rows += _tidy(
            rmse(Y[split.test], model.predict(pds.transform(Xs[split.test]))),
            ds,
            half_window=hw,
            rcond=rc,
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Streaming drift with continuous recalibration (simulation)
# --------------------------------------------------------------------------- #


def run_streaming(
    ds: CornDataset,
    slave: str = "mp5",
    n_steps: int = 240,
    drift_start: int = 60,
    drift_end: int = 180,
    n_std: int = 10,
    noise: float = 2e-4,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[int]]:
    """Simulate an instrument that drifts gradually from the master to ``slave``.

    The spectrum at time t is (1 - a_t) * master + a_t * slave (+ white noise) with a_t
    ramping linearly from 0 to 1 between ``drift_start`` and ``drift_end``. Two pipelines
    see the same stream:

    * ``static``  - the master model, never updated;
    * ``monitored`` - an EWMA chart on log(Q) triggers recalibration: re-measure
      ``n_std`` transfer standards on the instrument in its *current* state, fit PDS,
      and route subsequent spectra through it (and the monitor resets).
    """
    rng = np.random.default_rng(seed)
    split = standard_split(ds)
    Xm, Xs, Y = ds.X(MASTER), ds.X(slave), ds.y
    model = make_pipeline(PreprocessConfig(), pls_factory()).fit(Xm[split.train], Y[split.train])
    detector = PCADriftDetector().fit(Xm[split.train])
    monitor = EWMAMonitor(lam=0.2, L=3.0).fit(np.log(detector.cv_q_))
    std = split.train[kennard_stone(Xm[split.train], n_std)]

    alpha = np.clip((np.arange(n_steps) - drift_start) / (drift_end - drift_start), 0, 1)
    samples = rng.choice(split.test, size=n_steps)
    transfer = None
    recalibrations = []
    rows = []
    for t in range(n_steps):
        a, i = alpha[t], samples[t]
        x = ((1 - a) * Xm[i] + a * Xs[i] + noise * rng.standard_normal(Xm.shape[1]))[None]
        y_static = model.predict(x)[0]
        x_corr = transfer.transform(x) if transfer is not None else x
        y_mon = model.predict(x_corr)[0]
        _, q = detector.statistics(x_corr)
        z, alarm = monitor.update(float(np.log(q[0])))
        rows.append(
            {
                "t": t,
                "alpha": a,
                "sample": int(i),
                "log_q": float(np.log(q[0])),
                "ewma": z,
                "limit": monitor.limit_,
                "alarm": bool(alarm),
                **{
                    f"err_static_{p}": float(y_static[j] - Y[i, j])
                    for j, p in enumerate(ds.properties)
                },
                **{
                    f"err_monitored_{p}": float(y_mon[j] - Y[i, j])
                    for j, p in enumerate(ds.properties)
                },
            }
        )
        if alarm:
            # measure the transfer standards on the instrument as it is *now*
            Xstd_now = (
                (1 - a) * Xm[std] + a * Xs[std] + noise * rng.standard_normal((n_std, Xm.shape[1]))
            )
            transfer = PiecewiseDirectStandardization(half_window=5, rcond=0.1).fit(
                Xm[std], Xstd_now
            )
            monitor.reset()
            recalibrations.append(t)
    return pd.DataFrame(rows), recalibrations

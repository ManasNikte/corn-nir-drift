"""Static figures for the README (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# categorical slots in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def plot_spectra(wavelengths, spectra: dict[str, np.ndarray], out: Path, preprocessed=None):
    ncols = 2 if preprocessed is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.2), squeeze=False)
    for col, (data, title) in enumerate(
        [(spectra, "Raw absorbance, sample 1"), (preprocessed, "After default preprocessing")][
            :ncols
        ]
    ):
        ax = axes[0, col]
        for k, (inst, X) in enumerate(data.items()):
            ax.plot(wavelengths, X[0], color=SERIES[k], label=inst, lw=1.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_title(title, loc="left")
    axes[0, 0].legend(title="Instrument")
    fig.savefig(out)
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, out: Path):
    df = ablation[ablation.factorial].groupby(["model", "config"], as_index=False).rel_rmse.mean()
    order = df[df.model == "PLS"].sort_values("rel_rmse").config.tolist()
    models = sorted(df.model.unique(), key=lambda m: m != "PLS")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(order) + 1))
    h = 0.8 / len(models)
    y = np.arange(len(order))
    for k, m in enumerate(models):
        vals = df[df.model == m].set_index("config").reindex(order).rel_rmse
        ax.barh(y + k * h, vals, height=h - 0.04, color=SERIES[k], label=m)
    ax.set_yticks(y + h * (len(models) - 1) / 2, order)
    ax.invert_yaxis()
    ax.set_xlabel("RMSECV / property SD (mean over 4 properties, lower is better)")
    ax.set_title("Preprocessing ablation on instrument m5 (5-fold CV)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def plot_shift_heatmap(shift: pd.DataFrame, out: Path, model="PLS", preprocess="raw"):
    df = shift[(shift.model == model) & (shift.preprocess == preprocess)]
    props = df.property.unique()
    fig, axes = plt.subplots(1, len(props), figsize=(2.6 * len(props), 2.6))
    for ax, p in zip(axes, props, strict=True):
        m = df[df.property == p].pivot(index="train", columns="test", values="rmse")
        im = ax.imshow(np.log10(m.values), cmap="Blues")
        for (i, j), v in np.ndenumerate(m.values):
            light = im.norm(np.log10(v)) > 0.6
            ax.text(
                j,
                i,
                f"{v:.2g}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if light else INK,
            )
        ax.set_xticks(range(len(m.columns)), m.columns)
        ax.set_yticks(range(len(m.index)), m.index)
        ax.set_xlabel("test on")
        ax.set_title(p, loc="left")
        ax.grid(False)
    axes[0].set_ylabel("train on")
    fig.suptitle(f"{model} ({preprocess}) RMSEP across instruments", x=0.02, ha="left", y=1.04)
    fig.savefig(out)
    plt.close(fig)


def plot_transfer(err: pd.DataFrame, out: Path, slave="mp5"):
    df = err[err.slave == slave]
    props = df.property.unique()
    fig, axes = plt.subplots(1, len(props), figsize=(2.8 * len(props), 2.8), sharex=True)
    for ax, p in zip(axes, props, strict=True):
        d = df[df.property == p]
        for k, meth in enumerate(["PDS", "DS", "SBC"]):
            s = d[d.method == meth].sort_values("n_standards")
            ax.plot(s.n_standards, s.rmse, marker="o", ms=4, color=SERIES[k], label=meth)
        none = d[d.method == "none"].rmse.iloc[0]
        ref = d[d.method == "master (reference)"].rmse.iloc[0]
        ax.axhline(none, color=SERIES[3], ls="--", lw=1.2, label="no transfer")
        ax.axhline(ref, color=MUTED, ls=":", lw=1.2, label="master on master")
        ax.set_yscale("log")
        ax.set_title(p, loc="left")
        ax.set_xlabel("# transfer standards")
    axes[0].set_ylabel("RMSEP")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Calibration transfer m5 → {slave}", x=0.02, ha="left", y=1.04)
    fig.savefig(out)
    plt.close(fig)


def plot_streaming(stream: pd.DataFrame, recal: list[int], out: Path, prop="moisture"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(stream.t, stream.log_q, color=GRID, lw=1, label="log Q (per spectrum)")
    ax1.plot(stream.t, stream.ewma, color=SERIES[0], label="EWMA")
    ax1.axhline(stream.limit.iloc[0], color=SERIES[1], ls="--", lw=1.2, label="control limit")
    for t in recal:
        ax1.axvline(t, color=MUTED, lw=0.8, ls=":")
        ax2.axvline(t, color=MUTED, lw=0.8, ls=":")
    ax1.set_ylabel("log Q residual")
    ax1.legend(fontsize=7, loc="upper left", ncols=3)
    ax1.set_title("Drift monitor (dotted lines = alarm → recalibration with PDS)", loc="left")
    win = 10
    for k, kind in enumerate(["static", "monitored"]):
        e = stream[f"err_{kind}_{prop}"].pow(2).rolling(win, min_periods=1).mean().pow(0.5)
        ax2.plot(
            stream.t,
            e,
            color=SERIES[k],
            label={"static": "static model", "monitored": "monitored + recalibrated"}[kind],
        )
    ax2.set_yscale("log")
    ax2.set_ylabel(f"{prop} rolling RMSE")
    ax2.set_xlabel("measurement #")
    ax3 = ax2.twinx()
    ax3.fill_between(
        stream.t,
        stream.alpha,
        color=SERIES[3],
        alpha=0.15,
        lw=0,
        label="drift progress (m5 → mp5 mix)",
    )
    ax3.set_ylim(0, 4)
    ax3.set_yticks([])
    ax3.grid(False)
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2 + h3, l2 + l3, fontsize=7, loc="upper left")
    fig.savefig(out)
    plt.close(fig)

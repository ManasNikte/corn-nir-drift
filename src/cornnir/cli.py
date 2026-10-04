"""Command line entry point: ``cornnir {download,ablation,compare,shift,transfer,stream,all}``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import experiments as ex
from . import plots
from .data import download, load_corn
from .preprocess import DEFAULT_PREPROCESS


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4g")


def cmd_ablation(ds, out: Path, args):
    abl = ex.run_ablation(ds, include_cnn=not args.no_cnn, n_jobs=args.jobs)
    abl.to_csv(out / "ablation.csv", index=False)
    eff = ex.main_effects(abl)
    eff.to_csv(out / "ablation_main_effects.csv", index=False)
    summary = (
        abl.groupby(["model", "config", "property"])
        .rmse.mean()
        .unstack("property")
        .join(abl.groupby(["model", "config"]).rel_rmse.mean())
        .sort_values(["model", "rel_rmse"])
        .reset_index()
    )
    summary.to_csv(out / "ablation_summary.csv", index=False)
    (out / "ablation.md").write_text(
        "## RMSECV by preprocessing (m5, repeated 5-fold)\n\n"
        + _md(summary)
        + "\n\n## Main effects (% change in relative RMSECV when step is on)\n\n"
        + _md(eff)
        + "\n"
    )
    plots.plot_ablation(abl, out / "ablation.png")


def cmd_compare(ds, out: Path, args):
    cmp_ = ex.run_model_comparison(ds, n_jobs=args.jobs)
    cmp_.to_csv(out / "compare.csv", index=False)
    table = cmp_.pivot_table(index=["preprocess", "model"], columns="property", values="rmse")
    (out / "compare.md").write_text(
        "## Test RMSEP on m5 (Kennard-Stone 60/20 split)\n\n"
        + table.reset_index().to_markdown(index=False, floatfmt=".4g")
        + "\n"
    )


def cmd_shift(ds, out: Path, args):
    sh = ex.run_instrument_shift(ds, include_cnn=not args.no_cnn, n_jobs=args.jobs)
    sh.to_csv(out / "shift.csv", index=False)
    parts = []
    for (m, p), g in sh.groupby(["model", "preprocess"]):
        t = g.pivot_table(index="train", columns=["property", "test"], values="rmse")
        parts.append(f"### {m} / {p}\n\n" + t.round(4).to_markdown())
        plots.plot_shift_heatmap(sh, out / f"shift_{m}_{p}.png", m, p)
    (out / "shift.md").write_text(
        "## RMSEP: train instrument (rows) x test instrument\n\n" + "\n\n".join(parts) + "\n"
    )


def cmd_transfer(ds, out: Path, args):
    for pre_name, pre in [("raw", None), ("default", DEFAULT_PREPROCESS)]:
        err, flags = ex.run_transfer(ds, preprocess=pre)
        err.to_csv(out / f"transfer_{pre_name}.csv", index=False)
        flags.to_csv(out / f"drift_flags_{pre_name}.csv", index=False)
        for slave in ("mp5", "mp6"):
            plots.plot_transfer(err, out / f"transfer_{pre_name}_{slave}.png", slave)
        table = err.pivot_table(
            index=["slave", "method", "n_standards"], columns="property", values="rmse"
        ).reset_index()
        (out / f"transfer_{pre_name}.md").write_text(
            f"## Calibration transfer, master m5, preprocessing={pre_name}\n\n"
            + _md(table)
            + "\n\n## Fraction of test spectra flagged by drift detector\n\n"
            + _md(flags)
            + "\n"
        )
    grid = ex.run_pds_grid(ds)
    grid.to_csv(out / "pds_grid.csv", index=False)
    (out / "pds_grid.md").write_text(
        "## PDS hyper-parameters (m5→mp5, 10 standards, raw PLS)\n\n"
        + grid.pivot_table(index=["half_window", "rcond"], columns="property", values="rmse")
        .reset_index()
        .to_markdown(index=False, floatfmt=".4g")
        + "\n"
    )
    wl = ds.wavelengths
    pre = DEFAULT_PREPROCESS.build()
    plots.plot_spectra(
        wl,
        ds.spectra,
        out / "spectra.png",
        {k: pre.fit_transform(v) for k, v in ds.spectra.items()},
    )


def cmd_stream(ds, out: Path, args):
    stream, recal = ex.run_streaming(ds)
    stream.to_csv(out / "streaming.csv", index=False)
    plots.plot_streaming(stream, recal, out / "streaming.png")
    phases = {
        "before drift": stream.t < 60,
        "during drift": (stream.t >= 60) & (stream.t < 180),
        "after drift": stream.t >= 180,
    }
    rows = []
    for name, mask in phases.items():
        s = stream[mask]
        row = {"phase": name}
        for p in ds.properties:
            for kind in ("static", "monitored"):
                row[f"{p}_{kind}"] = float((s[f"err_{kind}_{p}"] ** 2).mean() ** 0.5)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "streaming_summary.csv", index=False)
    (out / "streaming.md").write_text(
        f"## Streaming drift simulation (m5 → mp5)\n\nRecalibrations at t = {recal}\n\n"
        + _md(summary)
        + "\n"
    )
    (out / "streaming_recalibrations.json").write_text(json.dumps(recal))


COMMANDS = {
    "ablation": cmd_ablation,
    "compare": cmd_compare,
    "shift": cmd_shift,
    "transfer": cmd_transfer,
    "stream": cmd_stream,
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cornnir", description=__doc__)
    ap.add_argument("command", choices=["download", *COMMANDS, "all"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (joblib)")
    ap.add_argument("--no-cnn", action="store_true", help="skip CNN runs (fast)")
    args = ap.parse_args(argv)

    if args.command == "download":
        print(download(args.data_dir))
        return
    ds = load_corn(args.data_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in COMMANDS if args.command == "all" else [args.command]:
        t0 = time.time()
        print(f"[cornnir] {name} ...", flush=True)
        COMMANDS[name](ds, args.out, args)
        print(f"[cornnir] {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

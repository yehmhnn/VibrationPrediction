"""Aggregate multi-seed TimesNet horizon results and compare deterministic baselines."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=project_root / "logs" / "baseline_horizon_sweep_within_105_0.csv",
    )
    parser.add_argument(
        "--timesnet",
        nargs="+",
        type=Path,
        default=[
            project_root / "logs" / f"timesnet_horizon_sweep_within_105_0_seed{s}.csv"
            for s in (42, 7, 123)
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "logs" / "horizon_sweep_summary_within_105_0.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=project_root / "figures" / "horizon_sweep_comparison_within_105_0.png",
    )
    args = parser.parse_args()

    baseline_rows = read_rows(args.baselines)
    timesnet_rows = [row for path in args.timesnet for row in read_rows(path)]
    horizons = sorted({int(row["H"]) for row in timesnet_rows})
    summary = []
    for horizon in horizons:
        candidates = [row for row in timesnet_rows if int(row["H"]) == horizon]
        maes = np.array([float(row["MAE"]) for row in candidates])
        rmses = np.array([float(row["RMSE"]) for row in candidates])
        runtimes = np.array([float(row["training_seconds"]) for row in candidates])
        summary.append(
            {
                "method": "timesnet",
                "H": horizon,
                "horizon_ms": 1000 * horizon / 12000,
                "seeds": ";".join(row["seed"] for row in candidates),
                "MAE_mean": maes.mean(),
                "MAE_sd": maes.std(ddof=1),
                "RMSE_mean": rmses.mean(),
                "RMSE_sd": rmses.std(ddof=1),
                "training_seconds_mean": runtimes.mean(),
            }
        )
        for method in ("mean", "autoregressive"):
            row = next(
                row
                for row in baseline_rows
                if int(row["H"]) == horizon and row["method"] == method
            )
            summary.append(
                {
                    "method": method,
                    "H": horizon,
                    "horizon_ms": float(row["horizon_ms"]),
                    "seeds": "",
                    "MAE_mean": float(row["MAE"]),
                    "MAE_sd": 0.0,
                    "RMSE_mean": float(row["RMSE"]),
                    "RMSE_sd": 0.0,
                    "training_seconds_mean": "",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    labels = {
        "mean": "Lookback mean",
        "autoregressive": "AR(512)",
        "timesnet": "TimesNet",
    }
    for method in ("mean", "autoregressive", "timesnet"):
        rows = sorted(
            (row for row in summary if row["method"] == method), key=lambda r: r["H"]
        )
        x = np.array([row["horizon_ms"] for row in rows])
        for axis, metric in zip(axes, ("MAE", "RMSE")):
            y = np.array([row[f"{metric}_mean"] for row in rows])
            sd = np.array([row[f"{metric}_sd"] for row in rows])
            axis.plot(x, y, marker="o", label=labels[method])
            if method == "timesnet":
                axis.fill_between(x, y - sd, y + sd, alpha=0.2)
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.set(xlabel="Forecast horizon [ms]", ylabel=f"{metric} [normalized signal]")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
    figure.suptitle("Within-record CWRU forecasting versus physical horizon")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=300)
    plt.close(figure)
    print(f"Saved horizon summary to {args.output}")
    print(f"Saved horizon comparison to {args.figure}")


if __name__ == "__main__":
    main()

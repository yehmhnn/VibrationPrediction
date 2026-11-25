"""Summarize cross-record TimesNet results with and without RPM conditioning."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_one(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def main():
    project_root = Path(__file__).resolve().parents[1]
    seeds = (42, 7, 123)
    groups = {
        "TimesNet": [
            project_root / "logs" / f"timesnet_L512_H96_seed{seed}_stride32_e5.csv"
            for seed in seeds
        ],
        "TimesNet + RPM": [
            project_root / "logs" / f"timesnet_rpm_L512_H96_seed{seed}.csv"
            for seed in seeds
        ],
    }
    rows = []
    for model, paths in groups.items():
        source = [load_one(path) for path in paths]
        for metric in ("MAE", "RMSE"):
            values = np.array([float(row[metric]) for row in source])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "seeds": ";".join(str(seed) for seed in seeds),
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                }
            )

    output = project_root / "logs" / "rpm_conditioning_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    reference = {
        "MAE": {"Mean": 0.767208, "AR(32)": 0.719477},
        "RMSE": {"Mean": 1.073380, "AR(32)": 1.023612},
    }
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        selected = [row for row in rows if row["metric"] == metric]
        names = [row["model"] for row in selected]
        means = [row["mean"] for row in selected]
        errors = [row["sample_sd"] for row in selected]
        axis.bar(names, means, yerr=errors, capsize=5)
        for name, value in reference[metric].items():
            axis.axhline(value, linestyle="--", label=name)
        axis.set_ylabel(f"{metric} [normalized signal]")
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=12)
        axis.legend()
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.suptitle("Cross-record CWRU: effect of RPM conditioning")
    figure_path = project_root / "figures" / "rpm_conditioning_comparison.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    print(f"Saved summary to {output}")
    print(f"Saved comparison to {figure_path}")


if __name__ == "__main__":
    main()

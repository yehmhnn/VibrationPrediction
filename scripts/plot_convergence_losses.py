"""Plot only new objective-matched histories; fails on missing/nonpositive data."""

import argparse
import csv
import math
import json
from pathlib import Path


def read_series(path, model):
    keys = (
        ["training_flow_matching_loss", "validation_flow_matching_loss"]
        if model == "tsflow"
        else ["training_forecast_MSE", "validation_forecast_MSE"]
    )
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty history")
    epochs = [int(r["epoch"]) for r in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError("Missing or repeated epochs")
    values = []
    for key in keys:
        if key not in rows[0]:
            raise ValueError("Missing objective-matched field: " + key)
        v = [float(r[key]) for r in rows]
        if any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError(
                (
                    "Log scale requires strictly positive finite loss; no "
                    "clipping/floors used"
                )
            )
        values.append(v)
    return epochs, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tsflow", "timesnet"], required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--protocol", choices=["single_105_0", "multi_record_B"], required=True
    )
    a = p.parse_args()
    if a.output.exists():
        p.error("Refusing to overwrite a figure")
    config_path = a.history.parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if (
            config.get("protocol", "single_105_0") != a.protocol
            or config.get("model") != a.model
        ):
            p.error("History/config protocol or model mismatch")
    epochs, values = read_series(a.history, a.model)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(epochs, values[0], label="Training (during optimization)")
    ax.plot(epochs, values[1], label="Validation (evaluation mode)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Flow Matching velocity MSE (normalized path units²)"
        if a.model == "tsflow"
        else "Forecast MSE (normalized signal units²)"
    )
    ax.set_title(
        (
            "Preliminary single-recording"
            if a.protocol == "single_105_0"
            else "Main Protocol B"
        )
        + " — "
        + ("TSFlow Flow Matching" if a.model == "tsflow" else "TimesNet forecast MSE")
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

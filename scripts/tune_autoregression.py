"""Select autoregressive hyperparameters on the manifest validation split."""

import argparse
import csv
from pathlib import Path

from run_baselines import evaluate, fit_autoregression
from sequence_data import ManifestSequenceDataset, default_cwru_dataset_dir


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_cwru_dataset_dir())
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument(
        "--orders", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--ridges", nargs="+", type=float, default=[1e-3, 1.0, 100.0])
    parser.add_argument("--fit-stride", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "logs" / "ar_validation_search.csv",
    )
    args = parser.parse_args()

    if max(args.orders) > args.lookback:
        raise ValueError("Every AR order must be no greater than the lookback")
    validation = ManifestSequenceDataset(
        args.dataset_dir,
        "validation",
        input_length=args.lookback,
        prediction_length=args.horizon,
        stride=args.horizon,
    )
    rows = []
    for order in args.orders:
        for ridge in args.ridges:
            coefficients, intercept, examples = fit_autoregression(
                args.dataset_dir,
                order=order,
                sample_stride=args.fit_stride,
                ridge=ridge,
            )
            metrics = evaluate(
                validation,
                ("autoregressive",),
                args.horizon,
                seasonal_period=1,
                ar_coefficients=coefficients,
                ar_intercept=intercept,
            )["autoregressive"]
            rows.append(
                {
                    "dataset": args.dataset_dir.name,
                    "selection_split": "validation",
                    "L": args.lookback,
                    "H": args.horizon,
                    "ar_order": order,
                    "ar_ridge": ridge,
                    "fit_stride": args.fit_stride,
                    "training_examples": examples,
                    **metrics,
                }
            )

    best = min(rows, key=lambda row: row["RMSE"])
    for row in rows:
        row["selected_by_rmse"] = row is best
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Selected AR({best['ar_order']}), ridge={best['ar_ridge']:g}: "
        f"validation MAE={best['MAE']:.6f}, RMSE={best['RMSE']:.6f}"
    )
    print(f"Saved {len(rows)} validation configurations to {args.output}")


if __name__ == "__main__":
    main()

"""Evaluate deterministic reference forecasts on a manifest-based test split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sequence_data import ManifestSequenceDataset, default_cwru_dataset_dir


def fit_autoregression(dataset_dir, order=128, sample_stride=4, ridge=1e-3):
    """Fit a one-step ridge autoregression using training records only.

    The normal equations are accumulated per physical record, so no regression
    example can cross a record boundary. Multi-step forecasts are recursive.
    """
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    if not rows:
        raise ValueError("No training records available for autoregression")
    if order <= 0 or sample_stride <= 0 or ridge < 0:
        raise ValueError(
            "AR order and sample stride must be positive; ridge must be non-negative"
        )

    # The final column represents an unpenalized intercept.
    gram = np.zeros((order + 1, order + 1), dtype=np.float64)
    cross = np.zeros(order + 1, dtype=np.float64)
    example_count = 0
    for row in rows:
        signal = np.asarray(
            np.load(dataset_dir / row["relative_path"]), dtype=np.float64
        )
        windows = np.lib.stride_tricks.sliding_window_view(signal, order + 1)[
            ::sample_stride
        ]
        predictors = windows[:, :-1]
        targets = windows[:, -1]
        design = np.column_stack((predictors, np.ones(predictors.shape[0])))
        gram += design.T @ design
        cross += design.T @ targets
        example_count += targets.size

    penalty = np.eye(order + 1, dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    parameters = np.linalg.solve(gram + penalty, cross)
    return parameters[:-1], float(parameters[-1]), example_count


def estimate_training_period(dataset_dir, minimum_period=2, maximum_period=96):
    """Estimate one integer seasonal period from training records only."""
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    if not rows:
        raise ValueError("No training records available for period estimation")

    periods = []
    for row in rows:
        signal = np.asarray(
            np.load(dataset_dir / row["relative_path"]), dtype=np.float64
        )
        signal = signal - signal.mean()
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        frequencies = np.fft.rfftfreq(signal.size)
        spectrum[0] = 0.0

        valid = frequencies > 0
        candidate_periods = np.divide(
            1.0,
            frequencies[valid],
            out=np.full(valid.sum(), np.inf),
            where=frequencies[valid] > 0,
        )
        valid_periods = (candidate_periods >= minimum_period) & (
            candidate_periods <= maximum_period
        )
        if not valid_periods.any():
            raise ValueError(f"No valid spectral period found for {row['record_id']}")
        valid_power = spectrum[valid][valid_periods]
        selected = candidate_periods[valid_periods][np.argmax(valid_power)]
        periods.append(int(np.clip(np.rint(selected), minimum_period, maximum_period)))

    return int(np.rint(np.median(periods))), periods


def predict(
    method, past, horizon, seasonal_period, ar_coefficients=None, ar_intercept=0.0
):
    """Return a deterministic forecast with shape (batch, horizon)."""
    if method == "zero":
        return np.zeros((past.shape[0], horizon), dtype=np.float64)
    if method == "mean":
        return np.repeat(past.mean(axis=1, keepdims=True), horizon, axis=1)
    if method == "last":
        return np.repeat(past[:, -1:], horizon, axis=1)
    if method == "seasonal":
        recent_period = past[:, -seasonal_period:]
        repetitions = int(np.ceil(horizon / seasonal_period))
        return np.tile(recent_period, (1, repetitions))[:, :horizon]
    if method == "autoregressive":
        if ar_coefficients is None:
            raise ValueError("Autoregressive prediction requires fitted coefficients")
        order = len(ar_coefficients)
        if past.shape[1] < order:
            raise ValueError(
                f"Lookback {past.shape[1]} is shorter than AR order {order}"
            )
        history = past[:, -order:].copy()
        forecast = np.empty((past.shape[0], horizon), dtype=np.float64)
        for lead in range(horizon):
            next_value = history @ ar_coefficients + ar_intercept
            forecast[:, lead] = next_value
            history[:, :-1] = history[:, 1:]
            history[:, -1] = next_value
        return forecast
    raise ValueError(f"Unknown baseline method: {method}")


def compute_metrics(target, forecast):
    error = forecast - target
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    target_std = np.std(target)
    nrmse = rmse / target_std if target_std > 0 else np.nan
    denominator = np.sum(np.abs(target))
    nd = np.sum(np.abs(error)) / denominator if denominator > 0 else np.nan
    target_flat = target.reshape(-1)
    forecast_flat = forecast.reshape(-1)
    if np.std(target_flat) > 0 and np.std(forecast_flat) > 0:
        correlation = np.corrcoef(target_flat, forecast_flat)[0, 1]
    else:
        correlation = np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "ND": nd,
        "correlation": correlation,
    }


def compute_lead_time_errors(
    dataset,
    methods,
    horizon,
    seasonal_period,
    ar_coefficients,
    ar_intercept,
    batch_size=1024,
):
    """Compute MAE and RMSE separately for every forecast lead time."""
    absolute_error = {method: np.zeros(horizon, dtype=np.float64) for method in methods}
    squared_error = {method: np.zeros(horizon, dtype=np.float64) for method in methods}
    count = 0
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        past = np.stack([item[0].numpy() for item in items]).astype(np.float64)
        target = np.stack([item[1].numpy() for item in items]).astype(np.float64)
        count += target.shape[0]
        for method in methods:
            forecast = predict(
                method, past, horizon, seasonal_period, ar_coefficients, ar_intercept
            )
            error = forecast - target
            absolute_error[method] += np.abs(error).sum(axis=0)
            squared_error[method] += (error**2).sum(axis=0)

    return {
        method: {
            "MAE": absolute_error[method] / count,
            "RMSE": np.sqrt(squared_error[method] / count),
        }
        for method in methods
    }


def save_lead_time_outputs(results, sampling_rate, output_csv, output_figure):
    horizon = len(next(iter(results.values()))["MAE"])
    rows = []
    for method, metrics in results.items():
        for lead_index in range(horizon):
            lead_samples = lead_index + 1
            rows.append(
                {
                    "method": method,
                    "lead_samples": lead_samples,
                    "lead_ms": 1000 * lead_samples / sampling_rate,
                    "MAE": metrics["MAE"][lead_index],
                    "RMSE": metrics["RMSE"][lead_index],
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    lead_ms = 1000 * np.arange(1, horizon + 1) / sampling_rate
    for method, metrics in results.items():
        axes[0].plot(lead_ms, metrics["MAE"], label=method)
        axes[1].plot(lead_ms, metrics["RMSE"], label=method)
    axes[0].set_ylabel("MAE [normalized signal]")
    axes[1].set_ylabel("RMSE [normalized signal]")
    for axis in axes:
        axis.set_xlabel("Forecast lead time [ms]")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
    figure.suptitle("CWRU deterministic baseline error by forecast lead time")
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_figure, dpi=300)
    plt.close(figure)


def evaluate(
    dataset,
    methods,
    horizon,
    seasonal_period,
    ar_coefficients,
    ar_intercept,
    batch_size=1024,
):
    totals = {method: {"targets": [], "forecasts": []} for method in methods}
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        past = np.stack([item[0].numpy() for item in items]).astype(np.float64)
        target = np.stack([item[1].numpy() for item in items]).astype(np.float64)
        for method in methods:
            totals[method]["targets"].append(target)
            totals[method]["forecasts"].append(
                predict(
                    method,
                    past,
                    horizon,
                    seasonal_period,
                    ar_coefficients,
                    ar_intercept,
                )
            )

    results = {}
    for method in methods:
        target = np.concatenate(totals[method]["targets"], axis=0)
        forecast = np.concatenate(totals[method]["forecasts"], axis=0)
        results[method] = compute_metrics(target, forecast)
    return results


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_cwru_dataset_dir())
    parser.add_argument("--lookback", type=int, default=192)
    parser.add_argument("--ar-order", type=int, default=128)
    parser.add_argument("--ar-fit-stride", type=int, default=4)
    parser.add_argument("--ar-ridge", type=float, default=1e-3)
    parser.add_argument("--horizons", nargs="+", type=int, default=[48, 96, 192, 336])
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride; defaults to the horizon to limit repeated target samples.",
    )
    parser.add_argument(
        "--output", type=Path, default=project_root / "logs" / "baseline_results.csv"
    )
    parser.add_argument(
        "--lead-output",
        type=Path,
        default=project_root / "logs" / "baseline_lead_time.csv",
    )
    parser.add_argument(
        "--lead-figure",
        type=Path,
        default=project_root / "figures" / "baseline_lead_time.png",
    )
    args = parser.parse_args()

    with (args.dataset_dir / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    sampling_rate = int(metadata["sampling_rate_hz"])
    seasonal_period, training_periods = estimate_training_period(
        args.dataset_dir, maximum_period=min(args.lookback, 96)
    )
    print(
        f"Training-only seasonal period: {seasonal_period} samples "
        f"(per-record estimates: {training_periods})"
    )
    if args.ar_order > args.lookback:
        raise ValueError("--ar-order cannot exceed --lookback")
    ar_coefficients, ar_intercept, ar_examples = fit_autoregression(
        args.dataset_dir,
        order=args.ar_order,
        sample_stride=args.ar_fit_stride,
        ridge=args.ar_ridge,
    )
    print(
        f"Training-only AR({args.ar_order}) fitted on {ar_examples} examples "
        f"(stride={args.ar_fit_stride}, ridge={args.ar_ridge:g})"
    )

    methods = ("zero", "mean", "last", "seasonal", "autoregressive")
    rows = []
    for horizon in args.horizons:
        stride = args.stride or horizon
        test_dataset = ManifestSequenceDataset(
            args.dataset_dir,
            "test",
            input_length=args.lookback,
            prediction_length=horizon,
            stride=stride,
        )
        results = evaluate(
            test_dataset,
            methods,
            horizon,
            seasonal_period,
            ar_coefficients,
            ar_intercept,
        )
        for method, metrics in results.items():
            row = {
                "dataset": args.dataset_dir.name,
                "split": "test",
                "record_ids": ";".join(test_dataset.record_ids),
                "method": method,
                "L": args.lookback,
                "H": horizon,
                "lookback_ms": 1000 * args.lookback / sampling_rate,
                "horizon_ms": 1000 * horizon / sampling_rate,
                "stride": stride,
                "seasonal_period": seasonal_period if method == "seasonal" else "",
                "ar_order": args.ar_order if method == "autoregressive" else "",
                "ar_ridge": args.ar_ridge if method == "autoregressive" else "",
                "num_windows": len(test_dataset),
                **metrics,
            }
            rows.append(row)
            print(
                f"H={horizon:3d} {method:8s}: MAE={metrics['MAE']:.6f}, "
                f"RMSE={metrics['RMSE']:.6f}, corr={metrics['correlation']:.6f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} baseline results to {args.output}")

    maximum_horizon = max(args.horizons)
    common_stride = args.stride or maximum_horizon
    lead_dataset = ManifestSequenceDataset(
        args.dataset_dir,
        "test",
        input_length=args.lookback,
        prediction_length=maximum_horizon,
        stride=common_stride,
    )
    lead_results = compute_lead_time_errors(
        lead_dataset,
        methods,
        maximum_horizon,
        seasonal_period,
        ar_coefficients,
        ar_intercept,
    )
    save_lead_time_outputs(
        lead_results,
        sampling_rate,
        args.lead_output,
        args.lead_figure,
    )
    print(
        f"Saved lead-time errors for {len(lead_dataset)} common test windows to "
        f"{args.lead_output} and {args.lead_figure}"
    )


if __name__ == "__main__":
    main()

"""Analyze periodicity and short-horizon predictability record by record."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def normalized_autocorrelation(values, max_lag):
    """Return overlap-corrected autocorrelation for non-negative lags."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    variance = np.dot(centered, centered) / centered.size
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Autocorrelation requires a signal with positive variance")
    max_lag = min(int(max_lag), centered.size - 1)
    correlation = scipy.signal.correlate(centered, centered, mode="full", method="fft")
    positive = correlation[centered.size - 1 : centered.size + max_lag]
    overlap = centered.size - np.arange(max_lag + 1)
    return positive / (overlap * variance)


def welch_spectrum(values, sampling_rate_hz, nperseg):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    frequencies, psd = scipy.signal.welch(
        values,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=min(int(nperseg), values.size),
        detrend="constant",
        scaling="density",
    )
    return frequencies, psd


def strongest_peaks(score, count, minimum_distance=1):
    peaks, _ = scipy.signal.find_peaks(score, distance=minimum_distance)
    if peaks.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(score[peaks])[::-1]
    return peaks[order[:count]]


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "datasets" / "processed" / "cwru_12k_de_ir007",
    )
    parser.add_argument("--max-lag", type=int, default=2000)
    parser.add_argument("--nperseg", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "logs" / "predictability_analysis.csv",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=project_root / "figures" / "predictability_analysis.png",
    )
    args = parser.parse_args()

    with (args.dataset_dir / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    sampling_rate = float(metadata["sampling_rate_hz"])
    with (args.dataset_dir / "manifest.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        manifest = list(csv.DictReader(stream))

    rows = []
    diagnostics = []
    for record in manifest:
        values = np.load(args.dataset_dir / record["relative_path"], mmap_mode="r")
        autocorrelation = normalized_autocorrelation(values, args.max_lag)
        frequencies, psd = welch_spectrum(values, sampling_rate, args.nperseg)

        # Ignore lag zero and require separated ACF maxima to avoid listing one broad peak.
        acf_peak_indices = (
            strongest_peaks(
                autocorrelation[1:],
                args.top_k,
                minimum_distance=3,
            )
            + 1
        )
        # Ignore DC. A 3-bin separation makes the ranked frequencies distinct.
        spectral_indices = strongest_peaks(psd[1:], args.top_k, minimum_distance=3) + 1

        for rank, lag in enumerate(acf_peak_indices, start=1):
            rows.append(
                {
                    "record_id": record["record_id"],
                    "split": record["split"],
                    "rpm": record["rpm"],
                    "analysis": "autocorrelation",
                    "rank": rank,
                    "frequency_hz": sampling_rate / lag,
                    "period_samples": lag,
                    "period_ms": 1000.0 * lag / sampling_rate,
                    "strength": autocorrelation[lag],
                }
            )
        for rank, index in enumerate(spectral_indices, start=1):
            frequency = frequencies[index]
            rows.append(
                {
                    "record_id": record["record_id"],
                    "split": record["split"],
                    "rpm": record["rpm"],
                    "analysis": "spectrum",
                    "rank": rank,
                    "frequency_hz": frequency,
                    "period_samples": sampling_rate / frequency,
                    "period_ms": 1000.0 / frequency,
                    "strength": psd[index],
                }
            )
        diagnostics.append((record, autocorrelation, frequencies, psd))

    fieldnames = [
        "record_id",
        "split",
        "rpm",
        "analysis",
        "rank",
        "frequency_hz",
        "period_samples",
        "period_ms",
        "strength",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "frequency_hz": f"{row['frequency_hz']:.6f}",
                    "period_samples": f"{row['period_samples']:.6f}",
                    "period_ms": f"{row['period_ms']:.6f}",
                    "strength": f"{row['strength']:.9g}",
                }
            )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for record, autocorrelation, frequencies, psd in diagnostics:
        label = f"{record['record_id']} ({record['split']}, {record['rpm']} RPM)"
        lags_ms = 1000.0 * np.arange(autocorrelation.size) / sampling_rate
        axes[0].plot(lags_ms[1:], autocorrelation[1:], label=label, linewidth=1)
        axes[1].semilogy(frequencies[1:], psd[1:], label=label, linewidth=1)
    axes[0].set(
        title="Record-level autocorrelation", xlabel="Lag [ms]", ylabel="Correlation"
    )
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].grid(True, alpha=0.3)
    axes[1].set(
        title="Record-level Welch spectra", xlabel="Frequency [Hz]", ylabel="PSD"
    )
    axes[1].set_xlim(0, sampling_rate / 2)
    axes[1].grid(True, which="both", alpha=0.3)
    for axis in axes:
        axis.legend(fontsize=8)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=250)
    plt.close(figure)

    print(f"Saved ranked periods to {args.output_csv}")
    print(f"Saved diagnostic figure to {args.output_figure}")


if __name__ == "__main__":
    main()

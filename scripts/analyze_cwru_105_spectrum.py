"""Reproducible spectral analysis of the chronological CWRU 105_0 splits."""

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, welch


FS = 12_000.0
NPERSEG = 8192
NOVERLAP = 4096
NFFT = 8192
TOP_K = 10
WINDOW = "hann"
DETREND = "constant"
SCALING = "density"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum(values):
    frequencies, psd = welch(
        values,
        fs=FS,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        detrend=DETREND,
        scaling=SCALING,
    )
    cumulative = cumulative_trapezoid(psd, frequencies, initial=0.0)
    cumulative /= cumulative[-1]
    return frequencies, psd, cumulative


def ranked_peaks(frequencies, psd):
    # Exclude DC and separate neighboring maxima by at least five FFT bins.
    indices, _ = find_peaks(psd[1:], distance=5)
    indices = indices + 1
    indices = indices[np.argsort(psd[indices])[::-1]][:TOP_K]
    return indices


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "datasets" / "processed" / "cwru_105_0_chronological"
    output_dir = project_root / "results" / "spectral_analysis_cwru_105_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), output_dir / Path(__file__).name)

    metadata_source = json.loads(
        (dataset_dir / "metadata.json").read_text(encoding="utf-8")
    )
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))

    diagnostics = {}
    peak_rows = []
    threshold_rows = []
    thresholds = (0.50, 0.90, 0.95)
    for row in manifest:
        split = row["split"]
        path = dataset_dir / row["relative_path"]
        values = np.asarray(np.load(path), dtype=np.float64)
        frequencies, psd, cumulative = spectrum(values)
        peak_indices = ranked_peaks(frequencies, psd)
        diagnostics[split] = (frequencies, psd, cumulative)
        for rank, index in enumerate(peak_indices, start=1):
            frequency = float(frequencies[index])
            peak_rows.append(
                {
                    "split": split,
                    "rank": rank,
                    "frequency_hz": frequency,
                    "period_seconds": 1.0 / frequency,
                    "period_ms": 1000.0 / frequency,
                    "samples_per_cycle": FS / frequency,
                    "psd": float(psd[index]),
                    "psd_relative_to_split_max": float(
                        psd[index] / psd[peak_indices[0]]
                    ),
                }
            )
        for threshold in thresholds:
            index = int(np.searchsorted(cumulative, threshold))
            threshold_rows.append(
                {
                    "split": split,
                    "cumulative_energy_fraction": threshold,
                    "threshold_frequency_hz": float(frequencies[index]),
                    "period_ms_at_threshold": 1000.0 / float(frequencies[index]),
                    "samples_per_cycle_at_threshold": FS / float(frequencies[index]),
                }
            )

    for filename, rows in (
        ("dominant_peaks.csv", peak_rows),
        ("cumulative_energy_thresholds.csv", threshold_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 9.5), constrained_layout=True)
    for split, (frequencies, psd, cumulative) in diagnostics.items():
        axes[0].plot(frequencies, psd, label=split, color=colors[split], linewidth=1.0)
        axes[1].semilogy(
            frequencies[1:],
            np.maximum(psd[1:], np.finfo(float).tiny),
            label=split,
            color=colors[split],
            linewidth=1.0,
        )
        axes[2].plot(
            frequencies, cumulative, label=split, color=colors[split], linewidth=1.2
        )
    axes[0].set(
        title="Welch power spectral density", ylabel="PSD [normalized signal²/Hz]"
    )
    axes[1].set(title="Welch PSD (log scale)", ylabel="PSD [normalized signal²/Hz]")
    axes[2].set(
        title="Cumulative spectral energy",
        xlabel="Frequency [Hz]",
        ylabel="Cumulative fraction",
        ylim=(0, 1.01),
    )
    for threshold in thresholds:
        axes[2].axhline(threshold, color="0.6", linestyle="--", linewidth=0.7)
    for axis in axes:
        axis.set_xlim(0, FS / 2)
        axis.grid(True, alpha=0.25)
        axis.legend()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"spectral_analysis.{extension}", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    for split, (frequencies, psd, _) in diagnostics.items():
        axis.semilogy(
            frequencies[1:],
            np.maximum(psd[1:], np.finfo(float).tiny),
            label=split,
            color=colors[split],
            linewidth=1.0,
        )
    axis.set(
        title="CWRU 105_0 spectral structure (log scale)",
        xlabel="Frequency [Hz]",
        ylabel="PSD [normalized signal²/Hz]",
        xlim=(0, FS / 2),
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"spectrum_log.{extension}", dpi=300)
    plt.close(figure)

    analysis_metadata = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CWRU Bearing Data Center",
        "record": "105_0",
        "processed_dataset": str(dataset_dir.resolve()),
        "sampling_rate_hz": FS,
        "signal_state": "training-region z-score normalized processed arrays",
        "normalization": metadata_source["normalization"],
        "training_normalization_mean": metadata_source["train_mean"],
        "training_normalization_std": metadata_source["train_std"],
        "boundaries": metadata_source["boundaries"],
        "gap_samples": metadata_source["gap_samples"],
        "inputs": [
            {
                "split": row["split"],
                "path": str((dataset_dir / row["relative_path"]).resolve()),
                "sha256": sha256(dataset_dir / row["relative_path"]),
                "source_start": int(row["source_start"]),
                "source_end_exclusive": int(row["source_end_exclusive"]),
                "num_samples": int(row["num_samples"]),
            }
            for row in manifest
        ],
        "spectral_method": "scipy.signal.welch",
        "welch": {
            "window": WINDOW,
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
            "nfft": NFFT,
            "detrend": DETREND,
            "scaling": SCALING,
            "frequency_resolution_hz": FS / NFFT,
        },
        "peak_selection": {
            "method": "scipy.signal.find_peaks; ranked by PSD",
            "dc_excluded": True,
            "minimum_distance_fft_bins": 5,
            "top_k": TOP_K,
        },
        "motivation": (
            "Assess CWRU timescales because inherited TSFlow frequency='H' maps to a "
            "24-sample hourly-benchmark period, not a CWRU-derived period."
        ),
        "decision_status": "No GP-frequency decision is made by this analysis.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2) + "\n", encoding="utf-8"
    )

    by_split_peaks = {
        split: [r for r in peak_rows if r["split"] == split] for split in diagnostics
    }
    by_split_thresholds = {
        split: [r for r in threshold_rows if r["split"] == split]
        for split in diagnostics
    }
    lines = ["# CWRU 105_0 spectral analysis", ""]
    for split in ("train", "validation", "test"):
        peak = by_split_peaks[split][0]
        threshold_text = ", ".join(
            f"{int(row['cumulative_energy_fraction'] * 100)}%: {row['threshold_frequency_hz']:.2f} Hz"
            for row in by_split_thresholds[split]
        )
        lines += [
            f"## {split.capitalize()}",
            "",
            f"- **Measured:** Highest ranked non-DC Welch peak: {peak['frequency_hz']:.2f} Hz.",
            f"- **Derived:** Period: {peak['period_ms']:.4f} ms; samples per cycle: {peak['samples_per_cycle']:.3f}.",
            f"- **Measured:** Cumulative-energy thresholds — {threshold_text}.",
            "",
        ]
    lines += [
        "## Cross-split observations",
        "",
        (
            "- **Measured:** Peak rankings and energy thresholds are reported "
            "separately for all splits in the CSV files."
        ),
        (
            "- **Interpretation:** Agreement of nearby peaks across splits supports "
            "persistent spectral components; disagreement indicates temporal variation "
            "within the record."
        ),
        (
            "- **Limitation:** Welch PSD measures power versus frequency but does not "
            "establish phase predictability or a unique GP prior period."
        ),
        (
            "- **Limitation:** Results describe one CWRU record and should not be "
            "generalized to other loads, faults, sensors, or datasets."
        ),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    traceability = """# Report traceability

- The inherited TSFlow setting `frequency="H"` maps to a 24-sample base period intended for hourly benchmark data with daily seasonality.
- No CWRU-specific analysis originally justified interpreting 24 vibration samples as a characteristic period.
- This analysis estimates Welch spectra independently for the chronological train, validation, and test regions of record `105_0`.
- `dominant_peaks.csv` and `cumulative_energy_thresholds.csv` contain the measured and derived timescales used for later experimental-design discussion.
- These results do not by themselves select or validate a GP frequency; that decision remains separate.
"""
    (output_dir / "traceability.md").write_text(traceability, encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()

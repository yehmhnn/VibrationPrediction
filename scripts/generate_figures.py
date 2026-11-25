"""Generate record-aware spectral screening figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal


OTTAWA_SAMPLING_RATE_HZ = 200_000


def compute_normalized_psd(signal, sampling_rate_hz, nperseg=8192):
    """Return a Welch spectrum normalized to unit total discrete power."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size < 2:
        raise ValueError("Signal too short for spectral analysis")
    segment_length = min(nperseg, signal.size)
    frequencies, psd = scipy.signal.welch(
        signal,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=segment_length,
        detrend="constant",
        scaling="density",
    )
    total = psd.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Signal has no finite positive spectral power")
    return frequencies, psd / total


def aggregate_record_spectra(records, sampling_rate_hz, nperseg=8192):
    """Average equal-weight normalized spectra without joining recordings."""
    spectra = []
    frequencies = None
    for signal in records:
        record_frequencies, spectrum = compute_normalized_psd(
            signal, sampling_rate_hz, nperseg=nperseg
        )
        if frequencies is None:
            frequencies = record_frequencies
        elif not np.array_equal(frequencies, record_frequencies):
            raise ValueError("Records produced incompatible frequency grids")
        spectra.append(spectrum)
    if not spectra:
        raise ValueError("No records supplied for spectral aggregation")
    return frequencies, np.mean(spectra, axis=0)


def load_manifest_records(dataset_dir):
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    records = [
        np.load(dataset_dir / row["relative_path"], mmap_mode="r") for row in rows
    ]
    return records, float(metadata["sampling_rate_hz"]), len(rows)


def load_ottawa_records(data_dir):
    paths = sorted(Path(data_dir).glob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"No Ottawa .mat files found in {data_dir}")
    records = []
    for path in paths:
        values = scipy.io.loadmat(path)
        if "Channel_1" not in values:
            raise KeyError(f"Channel_1 not found in {path}")
        records.append(values["Channel_1"].reshape(-1))
    return records, OTTAWA_SAMPLING_RATE_HZ, len(paths)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cwru-dir",
        type=Path,
        default=project_root / "datasets" / "processed" / "cwru_12k_de_ir007",
    )
    parser.add_argument(
        "--ottawa-dir", type=Path, default=project_root / "datasets" / "ottawa"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "figures" / "spectral_screening.png",
    )
    args = parser.parse_args()

    datasets = {
        "CWRU 12 kHz DE IR007": load_manifest_records(args.cwru_dir),
        "Ottawa Channel 1": load_ottawa_records(args.ottawa_dir),
    }

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, (name, (records, sampling_rate, count)) in zip(axes, datasets.items()):
        frequencies, spectrum = aggregate_record_spectra(records, sampling_rate)
        axis.semilogy(frequencies, np.maximum(spectrum, np.finfo(float).tiny))
        axis.set_title(f"{name}\n{count} records, {sampling_rate / 1000:g} kHz")
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel("Normalized spectral power per bin")
        axis.grid(True, which="both", linestyle="--", alpha=0.4)
        axis.set_xlim(0, sampling_rate / 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)
    print(f"Saved record-aware spectral figure to {args.output}")


if __name__ == "__main__":
    main()

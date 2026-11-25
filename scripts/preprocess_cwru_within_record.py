"""Create leakage-safe chronological regions from one CWRU recording."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io


SAMPLING_RATE_HZ = 12_000


def load_record(path):
    values = scipy.io.loadmat(path)
    signal_keys = sorted(key for key in values if key.endswith("DE_time"))
    rpm_keys = sorted(key for key in values if key.endswith("RPM"))
    if len(signal_keys) != 1 or len(rpm_keys) != 1:
        raise ValueError(
            f"Expected one DE_time and one RPM variable in {path}; "
            f"found {signal_keys} and {rpm_keys}"
        )
    signal = np.asarray(values[signal_keys[0]], dtype=np.float64).reshape(-1)
    rpm = int(np.asarray(values[rpm_keys[0]]).reshape(-1)[0])
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite samples in {path}")
    return signal, signal_keys[0], rpm


def chronological_boundaries(num_samples, gap_samples, train_fraction=0.6):
    available = num_samples - 2 * gap_samples
    if available <= 0:
        raise ValueError("Record is too short for two requested gaps")
    train_length = int(available * train_fraction)
    validation_length = (available - train_length) // 2
    test_length = available - train_length - validation_length
    train = (0, train_length)
    validation = (train[1] + gap_samples, train[1] + gap_samples + validation_length)
    test = (validation[1] + gap_samples, validation[1] + gap_samples + test_length)
    if test[1] != num_samples:
        raise AssertionError(
            "Chronological split does not cover the intended record extent"
        )
    return {"train": train, "validation": validation, "test": test}


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root
        / "datasets"
        / "cwru"
        / "12k_Drive_End_Bearing_Fault_Data"
        / "IR"
        / "007"
        / "105_0.mat",
    )
    parser.add_argument("--gap-samples", type=int, default=1200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "datasets" / "processed" / "cwru_105_0_chronological",
    )
    args = parser.parse_args()

    signal, source_key, rpm = load_record(args.source)
    boundaries = chronological_boundaries(signal.size, args.gap_samples)
    train_start, train_end = boundaries["train"]
    train_mean = float(signal[train_start:train_end].mean())
    train_std = float(signal[train_start:train_end].std())
    if train_std <= 0:
        raise ValueError("Training region has zero standard deviation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (start, end) in boundaries.items():
        normalized = ((signal[start:end] - train_mean) / train_std).astype(np.float32)
        output_name = f"105_0_{split}.npy"
        np.save(args.output_dir / output_name, normalized)
        rows.append(
            {
                "record_id": f"105_0_{split}",
                "parent_record_id": "105_0",
                "split": split,
                "relative_path": output_name,
                "source_path": str(args.source.relative_to(project_root)),
                "source_key": source_key,
                "source_start": start,
                "source_end_exclusive": end,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "num_samples": normalized.size,
                "duration_seconds": f"{normalized.size / SAMPLING_RATE_HZ:.6f}",
                "sensor": "drive_end_accelerometer",
                "condition": "inner_race_fault",
                "fault_diameter_inch": "0.007",
                "load_id": 0,
                "rpm": rpm,
            }
        )

    with (args.output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": "CWRU Bearing Data Center",
        "subset": "105_0 chronological regions",
        "parent_record_id": "105_0",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "rpm": rpm,
        "gap_samples": args.gap_samples,
        "gap_seconds": args.gap_samples / SAMPLING_RATE_HZ,
        "boundaries": boundaries,
        "normalization": "z-score using chronological training region only",
        "train_mean": train_mean,
        "train_std": train_std,
        "purpose": (
            "within-record raw-waveform predictability without overlapping split "
            "regions"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared chronological dataset in {args.output_dir}")
    for row in rows:
        print(
            f"{row['split']:10s}: [{row['source_start']}, {row['source_end_exclusive']}), "
            f"{row['num_samples']} samples ({row['duration_seconds']} s)"
        )


if __name__ == "__main__":
    main()

"""Prepare a record-aware CWRU baseline dataset.

The first baseline deliberately uses one internally consistent acquisition family:
12 kHz drive-end bearing-fault experiments, inner-race fault, 0.007 inch.
Each source recording remains a separate NumPy file so windows can never cross
recording or train/validation/test boundaries.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import scipy.io


SAMPLING_RATE_HZ = 12_000
DEFAULT_RELATIVE_INPUT = Path("12k_Drive_End_Bearing_Fault_Data/IR/007")
DEFAULT_SPLITS = {"0": "train", "1": "train", "2": "validation", "3": "test"}


def extract_unique_de_signal(mat_path):
    """Return the sole drive-end acceleration series in one CWRU file."""
    values = scipy.io.loadmat(mat_path)
    keys = sorted(key for key in values if key.endswith("DE_time"))
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one DE_time variable in {mat_path}, found {keys}"
        )
    signal = np.asarray(values[keys[0]], dtype=np.float64).reshape(-1)
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite samples found in {mat_path}")

    rpm_keys = sorted(key for key in values if key.endswith("RPM"))
    rpm = (
        int(np.asarray(values[rpm_keys[0]]).reshape(-1)[0])
        if len(rpm_keys) == 1
        else None
    )
    return signal, keys[0], rpm


def parse_load_id(path):
    match = re.search(r"_(\d+)\.mat$", path.name)
    if match is None:
        raise ValueError(f"Could not infer load ID from {path.name}")
    return match.group(1)


def preprocess(data_root=None, output_dir=None):
    project_root = Path(__file__).resolve().parents[1]
    cwru_root = Path(data_root) if data_root else project_root / "datasets" / "cwru"
    source_dir = cwru_root / DEFAULT_RELATIVE_INPUT
    destination = (
        Path(output_dir)
        if output_dir
        else project_root / "datasets" / "processed" / "cwru_12k_de_ir007"
    )

    source_files = sorted(source_dir.glob("*.mat"))
    if not source_files:
        raise FileNotFoundError(f"No CWRU .mat files found in {source_dir}")

    records = []
    for source_path in source_files:
        load_id = parse_load_id(source_path)
        if load_id not in DEFAULT_SPLITS:
            raise ValueError(f"No split assigned to load ID {load_id} in {source_path}")
        signal, source_key, rpm = extract_unique_de_signal(source_path)
        records.append(
            {
                "source_path": source_path,
                "record_id": source_path.stem,
                "load_id": load_id,
                "split": DEFAULT_SPLITS[load_id],
                "source_key": source_key,
                "rpm": rpm,
                "signal": signal,
            }
        )

    train_samples = np.concatenate(
        [r["signal"] for r in records if r["split"] == "train"]
    )
    train_mean = float(train_samples.mean())
    train_std = float(train_samples.std())
    if train_std == 0:
        raise ValueError("Training data have zero standard deviation")

    destination.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for record in records:
        normalized = ((record["signal"] - train_mean) / train_std).astype(np.float32)
        output_path = destination / f"{record['record_id']}.npy"
        np.save(output_path, normalized)
        manifest_rows.append(
            {
                "record_id": record["record_id"],
                "split": record["split"],
                "relative_path": output_path.name,
                "source_path": str(record["source_path"].relative_to(project_root)),
                "source_key": record["source_key"],
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "num_samples": normalized.size,
                "duration_seconds": f"{normalized.size / SAMPLING_RATE_HZ:.6f}",
                "sensor": "drive_end_accelerometer",
                "condition": "inner_race_fault",
                "fault_diameter_inch": "0.007",
                "load_id": record["load_id"],
                "rpm": record["rpm"] if record["rpm"] is not None else "",
            }
        )

    fieldnames = list(manifest_rows[0])
    with (destination / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "dataset": "CWRU Bearing Data Center",
        "subset": "12 kHz drive-end, inner-race fault, 0.007 inch",
        "target_channel": "DE_time",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "normalization": "z-score using training recordings only",
        "train_mean": train_mean,
        "train_std": train_std,
        "split_policy": DEFAULT_SPLITS,
        "warning": (
            "Load/RPM changes with split; test performance measures cross-load "
            "generalization."
        ),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Prepared {len(records)} separate CWRU records in {destination}")
    for row in manifest_rows:
        print(
            f"  {row['record_id']}: {row['split']}, {row['num_samples']} samples, "
            f"{row['duration_seconds']} s, {row['rpm']} RPM"
        )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    preprocess(data_root=args.data_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

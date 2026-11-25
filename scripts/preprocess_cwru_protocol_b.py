"""Explicit record-level preprocessing, fresh output only; no model execution."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from cwru_protocols import PROTOCOL_B, PROTOCOL_DATASETS, AUTHORITY, ROOT


def preprocess_assignments(data_root, output, specs, metadata_authority=AUTHORITY):
    """Validate recordings, fit training-only normalization and write artifacts."""
    output = Path(output)
    data_root = Path(data_root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite processed dataset: " + str(output))
    if len({record["record_id"] for record in specs}) != len(specs):
        raise ValueError("Recording assigned more than once")
    if set(record["split"] for record in specs) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test assignments")
    if len({record["source_relative_path"] for record in specs}) != len(specs):
        raise ValueError("Duplicate source recording")
    if any(
        Path(record["record_id"]).name != record["record_id"]
        or record["record_id"] in [".", ".."]
        for record in specs
    ):
        raise ValueError("Unsafe recording identifier")
    records = []
    for spec in specs:
        rel = Path(spec["source_relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe source path")
        source_path = data_root / rel
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        mat_data = loadmat(source_path)
        drive_end_keys = [k for k in mat_data if k.endswith("DE_time")]
        if drive_end_keys != [spec["source_key"]]:
            raise ValueError("Unexpected drive-end channel in " + str(source_path))
        if (
            spec["sampling_rate_hz"] != 12000
            or spec["sensor"] != "drive_end_accelerometer"
        ):
            raise ValueError("Acquisition family mismatch")
        signal = np.asarray(mat_data[spec["source_key"]], dtype=np.float64).reshape(-1)
        if signal.size != spec["num_samples"] or signal.size < 608:
            raise ValueError("Unexpected signal length")
        if not np.isfinite(signal).all():
            raise ValueError("Nonfinite source signal")
        rpm_keys = [k for k in mat_data if k.endswith("RPM")]
        if (
            len(rpm_keys) != 1
            or float(np.asarray(mat_data[rpm_keys[0]]).ravel()[0]) != spec["rpm"]
        ):
            raise ValueError("Missing/inconsistent RPM")
        records.append(
            (
                spec,
                source_path,
                signal,
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
        )
    training_samples = np.concatenate(
        [
            signal
            for record, source_path, signal, source_digest in records
            if record["split"] == "train"
        ]
    )
    mean, std = float(training_samples.mean()), float(training_samples.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid training standard deviation")
    normalization = {
        "formula": "(x - train_mean) / train_std",
        "estimator": "sample-weighted population mean/std; float64, ddof=0",
        "training_record_ids": [
            record["record_id"]
            for record, source_path, signal, source_digest in records
            if record["split"] == "train"
        ],
        "training_sample_count": int(training_samples.size),
        "train_mean": mean,
        "train_std": std,
        "source_sha256": {
            str(source_path.resolve()): source_digest
            for record, source_path, signal, source_digest in records
            if record["split"] == "train"
        },
        "uses_validation_or_test": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    normalization_path = output / "normalization.json"
    normalization_path.write_text(json.dumps(normalization, indent=2) + "\n")
    norm_hash = hashlib.sha256(normalization_path.read_bytes()).hexdigest()
    rows = []
    hashes = {}
    for record, source_path, signal, source_digest in records:
        name = record["record_id"] + ".npy"
        np.save(output / name, ((signal - mean) / std).astype(np.float32))
        hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
        rows.append(
            {
                **record,
                "fault_type": record["condition"],
                "parent_record_id": record["record_id"],
                "source_path": str(source_path.resolve()),
                "source_start": 0,
                "source_end_exclusive": signal.size,
                "duration_seconds": signal.size / 12000,
                "relative_path": name,
                "normalization_provenance": "normalization.json",
                "normalization_sha256": norm_hash,
                "source_sha256": source_digest,
                "metadata_authority": metadata_authority,
                "metadata_verification": (
                    "channel/RPM/count from MAT; rate/fault/load/position from "
                    "acquisition table"
                ),
            }
        )
    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "multi_record_B",
        "dataset": "CWRU",
        "sampling_rate_hz": 12000,
        "normalization": "z-score using training recordings only",
        "train_mean": mean,
        "train_std": std,
        "normalization_provenance": "normalization.json",
        "normalization_sha256": norm_hash,
        "metadata_authority": metadata_authority,
        "split_record_ids": {
            s: [record["record_id"] for record in rows if record["split"] == s]
            for s in ["train", "validation", "test"]
        },
        "windowing": {"L": 512, "H": 96, "train_stride": 32, "eval_stride": 96},
        "processed_sha256": hashes,
        "manifest_sha256": hashlib.sha256(
            (output / "manifest.csv").read_bytes()
        ).hexdigest(),
        "scope": "joint load/RPM shift; seen fault types and diameter; separate records",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows, normalization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/cwru")
    p.add_argument(
        "--output-dir", type=Path, default=PROTOCOL_DATASETS["multi_record_B"]
    )
    a = p.parse_args()
    rows, n = preprocess_assignments(a.data_root, a.output_dir, PROTOCOL_B)
    print(
        "Prepared",
        len(rows),
        "records; training normalization:",
        n["train_mean"],
        n["train_std"],
    )
    for s in ["train", "validation", "test"]:
        rs = [r for r in rows if r["split"] == s]
        print(
            s,
            "samples",
            sum(r["num_samples"] for r in rs),
            "windows",
            sum(
                (r["num_samples"] - 608) // (32 if s == "train" else 96) + 1 for r in rs
            ),
        )


if __name__ == "__main__":
    main()

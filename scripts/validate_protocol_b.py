"""Read-only checks of the approved processed protocol and provenance."""

import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from cwru_protocols import PROTOCOL_B


def verify_protocol_b(dataset_dir):
    root = Path(dataset_dir)
    meta = json.loads((root / "metadata.json").read_text())
    if (
        meta.get("protocol") != "multi_record_B"
        or meta.get("sampling_rate_hz") != 12000
    ):
        raise ValueError("Wrong protocol")
    if meta.get("normalization") != "z-score using training recordings only":
        raise ValueError("Wrong normalization")
    if meta.get("windowing") != {
        "L": 512,
        "H": 96,
        "train_stride": 32,
        "eval_stride": 96,
    }:
        raise ValueError("Window configuration changed")
    if hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest() != meta.get(
        "manifest_sha256"
    ):
        raise ValueError("Manifest hash mismatch")
    with (root / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    expected = {r["record_id"]: r for r in PROTOCOL_B}
    if len(rows) != 12 or {r["record_id"] for r in rows} != set(expected):
        raise ValueError("Unexpected/duplicate recording IDs")
    norm = root / "normalization.json"
    n = json.loads(norm.read_text())
    norm_hash = hashlib.sha256(norm.read_bytes()).hexdigest()
    train_ids = [r["record_id"] for r in PROTOCOL_B if r["split"] == "train"]
    if (
        set(n.get("training_record_ids", [])) != set(train_ids)
        or len(n["training_record_ids"]) != 6
        or n.get("training_sample_count") != 731654
        or n.get("uses_validation_or_test") is not False
    ):
        raise ValueError("Invalid normalization provenance")
    if (
        norm_hash != meta.get("normalization_sha256")
        or n["train_mean"] != meta["train_mean"]
        or n["train_std"] != meta["train_std"]
        or not np.isfinite(n["train_mean"])
        or not np.isfinite(n["train_std"])
        or n["train_std"] <= 0
    ):
        raise ValueError("Invalid normalization values/hash")
    paths = []
    source_paths = []
    for row in rows:
        e = expected[row["record_id"]]
        for key in [
            "split",
            "source_key",
            "condition",
            "sensor",
            "outer_race_position",
        ]:
            if row[key] != str(e[key]):
                raise ValueError("Metadata mismatch: " + key)
        for key in [
            "load_id",
            "rpm",
            "sampling_rate_hz",
            "num_samples",
            "fault_diameter_inch",
        ]:
            if float(row[key]) != e[key]:
                raise ValueError("Metadata mismatch: " + key)
        if (
            row["parent_record_id"] != row["record_id"]
            or int(row["source_start"]) != 0
            or int(row["source_end_exclusive"]) != e["num_samples"]
        ):
            raise ValueError("Whole-record boundaries changed")
        rel = Path(row["relative_path"])
        path = (root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or path.parent != root.resolve():
            raise ValueError("Invalid array path")
        paths.append(path)
        source_paths.append(Path(row["source_path"]).resolve())
        x = np.load(path, mmap_mode="r")
        if x.ndim != 1 or x.size != e["num_samples"] or not np.isfinite(x).all():
            raise ValueError("Invalid processed signal")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != meta["processed_sha256"][str(rel)]
        ):
            raise ValueError("Array hash mismatch")
        if (
            row["normalization_provenance"] != "normalization.json"
            or row["normalization_sha256"] != norm_hash
        ):
            raise ValueError("Row normalization mismatch")
    if len(set(paths)) != 12 or len(set(source_paths)) != 12:
        raise ValueError("Duplicate recording paths")
    if meta["split_record_ids"] != {
        s: [r["record_id"] for r in rows if r["split"] == s]
        for s in ["train", "validation", "test"]
    }:
        raise ValueError("Split metadata mismatch")
    return meta, rows

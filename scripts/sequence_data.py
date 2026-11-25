"""Record-aware datasets for vibration forecasting."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ManifestSequenceDataset(Dataset):
    """Create windows within records belonging to one declared split.

    No window can cross a source-record boundary. The manifest is expected to
    reference arrays already normalized using training-only statistics.
    """

    def __init__(
        self, dataset_dir, split, input_length=192, prediction_length=96, stride=1
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.input_length = int(input_length)
        self.prediction_length = int(prediction_length)
        self.stride = int(stride)
        if self.input_length <= 0 or self.prediction_length <= 0 or self.stride <= 0:
            raise ValueError(
                "input_length, prediction_length, and stride must be positive"
            )

        manifest_path = self.dataset_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not rows:
            raise ValueError(f"Manifest contains no records for split {split!r}")

        self.records = []
        self.window_index = []
        total_length = self.input_length + self.prediction_length
        for row in rows:
            path = self.dataset_dir / row["relative_path"]
            signal = np.load(path, mmap_mode="r")
            if signal.ndim != 1:
                raise ValueError(f"Expected a 1D signal in {path}, got {signal.shape}")
            record_index = len(self.records)
            self.records.append((row, signal))
            last_start = signal.size - total_length
            if last_start >= 0:
                self.window_index.extend(
                    (record_index, start)
                    for start in range(0, last_start + 1, self.stride)
                )
        if not self.window_index:
            raise ValueError(
                f"No {split} windows fit L={self.input_length}, H={self.prediction_length}"
            )

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, index):
        record_index, start = self.window_index[index]
        _, signal = self.records[record_index]
        boundary = start + self.input_length
        x = np.array(signal[start:boundary], dtype=np.float32, copy=True)
        y = np.array(
            signal[boundary : boundary + self.prediction_length],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def record_ids(self):
        return [row["record_id"] for row, _ in self.records]


class ManifestRPMSequenceDataset(ManifestSequenceDataset):
    """Return waveform windows with a training-normalized RPM condition."""

    def __init__(self, *args, rpm_mean, rpm_std, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpm_mean = float(rpm_mean)
        self.rpm_std = float(rpm_std)
        if (
            not np.isfinite(self.rpm_mean)
            or not np.isfinite(self.rpm_std)
            or self.rpm_std <= 0
        ):
            raise ValueError(
                "RPM normalization requires finite mean and positive standard deviation"
            )
        for row, _ in self.records:
            if not row.get("rpm"):
                raise ValueError(f"Record {row['record_id']} has no RPM metadata")

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        record_index, _ = self.window_index[index]
        row, _ = self.records[record_index]
        normalized_rpm = (float(row["rpm"]) - self.rpm_mean) / self.rpm_std
        return x, torch.tensor([normalized_rpm], dtype=torch.float32), y


def default_cwru_dataset_dir():
    return (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "cwru_12k_de_ir007"
    )

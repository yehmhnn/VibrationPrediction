"""Keep record identities out of model inputs; pooled/equal-record/fault metrics."""

from collections import defaultdict
import math
import torch
from torch.utils.data import Dataset


class IdentifiedDataset(Dataset):
    def __init__(self, dataset, flow=False):
        self.dataset = dataset
        self.flow = flow
        self.windows = dataset.windows if flow else dataset
        self.record_metadata = {r["record_id"]: r for r, x in self.windows.records}

    def __len__(self):
        return len(self.dataset)

    @property
    def record_ids(self):
        return self.dataset.record_ids

    def __getitem__(self, index):
        """Attach the record identity and absolute forecast origin to a window."""
        record_index, start = self.windows.window_index[index]
        record, _ = self.windows.records[record_index]
        record_id = record["record_id"]
        origin = (
            int(record.get("source_start") or 0) + start + self.windows.input_length
        )
        item = self.dataset[index]
        if self.flow:
            return {**item, "record_id": record_id, "source_forecast_start": origin}
        return (*item, record_id, origin)


def metadata(loader):
    return getattr(loader.dataset, "record_metadata", {})


class RecordMetrics:
    def __init__(self, record_metadata=None):
        self.metadata = record_metadata or {}
        self.sums = defaultdict(lambda: [0.0, 0.0, 0, 0])

    def update(self, pred, target, record_ids):
        """Accumulate absolute and squared errors for each recording."""
        if (
            pred.shape != target.shape
            or pred.ndim != 2
            or len(record_ids) != len(target)
        ):
            raise ValueError("Invalid recording metric batch")
        error = (
            pred.detach().to(dtype=torch.float64)
            - target.detach().to(dtype=torch.float64)
        ).cpu()
        if not torch.isfinite(error).all():
            raise ValueError("Nonfinite evaluation error")
        for i, record_id in enumerate(record_ids):
            record_totals = self.sums[str(record_id)]
            record_totals[0] += error[i].abs().sum().item()
            record_totals[1] += error[i].square().sum().item()
            record_totals[2] += error[i].numel()
            record_totals[3] += 1

    @staticmethod
    def metric(a):
        if a[2] <= 0:
            raise ValueError("No target elements")
        return {
            "MAE": a[0] / a[2],
            "MSE": a[1] / a[2],
            "RMSE": math.sqrt(a[1] / a[2]),
            "num_target_elements": a[2],
            "num_windows": a[3],
        }

    def result(self):
        """Return pooled, equal-record, per-record and per-fault metrics."""
        if not self.sums:
            raise ValueError("No evaluated records")
        per_record = {}
        pooled = [0.0, 0.0, 0, 0]
        fault_totals = defaultdict(lambda: [0.0, 0.0, 0, 0])
        for record_id, record_totals in self.sums.items():
            record_metadata = self.metadata.get(record_id, {})
            fault = record_metadata.get("condition", "unspecified")
            load = record_metadata.get("load_id")
            rpm = record_metadata.get("rpm")
            per_record[record_id] = {
                **self.metric(record_totals),
                "fault_type": fault,
                "load": int(load) if load not in [None, ""] else None,
                "RPM": float(rpm) if rpm not in [None, ""] else None,
                "parent_record_id": record_metadata.get("parent_record_id", record_id),
            }
            for i in range(4):
                pooled[i] += record_totals[i]
                fault_totals[fault][i] += record_totals[i]
        return {
            "pooled": self.metric(pooled),
            "equal_record": {
                "MAE": sum(r["MAE"] for r in per_record.values()) / len(per_record),
                "RMSE": sum(r["RMSE"] for r in per_record.values()) / len(per_record),
                "definition": (
                    "arithmetic mean of per-record MAEs/RMSEs; RMSE is not sqrt(mean "
                    "record MSE)"
                ),
                "num_records": len(per_record),
            },
            "per_record": per_record,
            "per_fault_type": {
                f: self.metric(record_totals)
                for f, record_totals in fault_totals.items()
            },
        }

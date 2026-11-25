"""Adapt record-aware forecast windows to TSFlow's batch contract."""

import torch
from torch.utils.data import Dataset

from sequence_data import ManifestSequenceDataset


class TSFlowManifestDataset(Dataset):
    """Expose manifest windows as the dictionary expected by TSFlowCond.

    The stored arrays were normalized with training-only statistics, so the
    adapter uses TSFlow's identity normalization contract (`mean=0`).
    """

    def __init__(
        self,
        dataset_dir,
        split,
        context_length,
        prediction_length,
        stride=1,
    ):
        self.windows = ManifestSequenceDataset(
            dataset_dir=dataset_dir,
            split=split,
            input_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        past, future = self.windows[index]
        return {
            "past_target": past,
            "future_target": future,
            "past_observed_values": torch.ones_like(past),
            "future_observed_values": torch.ones_like(future),
            "mean": torch.zeros(1, dtype=past.dtype),
        }

    @property
    def record_ids(self):
        return self.windows.record_ids

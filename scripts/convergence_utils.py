"Objective-matched validation and practical stabilization; never proves convergence."

import csv
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from record_evaluation import RecordMetrics, metadata

FLOW_FIELDS = [
    "epoch",
    "optimizer_step",
    "training_flow_matching_loss",
    "validation_flow_matching_loss",
    "validation_forecast_RMSE",
    "validation_forecast_MAE",
    "elapsed_training_seconds",
    "optimization_seconds",
    "validation_seconds",
    "assessment",
]
TIMESNET_FIELDS = [
    "epoch",
    "optimizer_step",
    "training_forecast_MSE",
    "validation_forecast_MSE",
    "validation_forecast_RMSE",
    "validation_forecast_MAE",
    "elapsed_training_seconds",
    "optimization_seconds",
    "validation_seconds",
    "assessment",
]


@contextmanager
def evaluation_context(model, device, seed):
    """Fixed draws, isolated Python/NumPy/Torch RNG, no grads; restore mode/RNG."""
    state = model.training
    py_state, np_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            model.eval()
            yield
    finally:
        model.train(state)
        random.setstate(py_state)
        np.random.set_state(np_state)


def move(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    return tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)


def flow_objective(model, batch):
    """Same extraction/p_losses as original driver, including context + future."""
    x1, x0, _, _, _, features = model._extract_features(batch)
    t = torch.rand((x1.shape[0], 1), device=x1.device)
    loss = model.p_losses(x1, x0, t, features)
    return loss, x1.numel()


def validation_flow_loss(model, loader, device, seed, draws=4):
    if draws < 1:
        raise ValueError("draws must be positive")
    total, count = 0.0, 0
    for draw in range(draws):
        with evaluation_context(model, device, seed + draw):
            for batch in loader:
                loss, n = flow_objective(model, move(batch, device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation Flow Matching loss")
                total += loss.item() * n
                count += n
    if not count:
        raise ValueError("Empty validation loader")
    return total / count


def validation_forecast(
    model, loader, device, seed, flow=False, num_samples=8, arrays=False, prefixes=()
):
    """Evaluate point forecasts with optional prefix metrics and arrays."""
    total_abs, total_sq, count = 0.0, 0.0, 0
    targets, predictions = [], []
    ids_saved, origins_saved = [], []
    records = RecordMetrics(metadata(loader))
    prefix_metrics = {k: RecordMetrics(metadata(loader)) for k in prefixes}
    previous_samples = getattr(model, "num_samples", None)
    try:
        with evaluation_context(model, device, seed):
            if flow:
                model.num_samples = num_samples
            for batch in loader:
                batch = move(batch, device)
                if flow:
                    target = batch["future_target"]
                    prediction = model(
                        batch["past_target"],
                        batch["past_observed_values"],
                        batch["mean"],
                    ).mean(dim=1)
                else:
                    x, target = batch[:2]
                    prediction = model(x.float())
                if prediction.shape != target.shape:
                    raise ValueError("Prediction/target shape mismatch")
                record_ids = (
                    batch.get("record_id", ["aggregate"] * len(target))
                    if flow
                    else batch[2]
                    if len(batch) > 2
                    else ["aggregate"] * len(target)
                )
                origins = (
                    batch.get("source_forecast_start")
                    if flow
                    else batch[3]
                    if len(batch) > 3
                    else None
                )
                records.update(prediction, target, record_ids)
                for k, acc in prefix_metrics.items():
                    if not 0 < k <= target.shape[1]:
                        raise ValueError("Invalid cumulative prefix")
                    acc.update(prediction[:, :k], target[:, :k], record_ids)
                if arrays:
                    ids_saved.extend(record_ids)
                    origins_saved.extend(
                        origins.cpu().tolist()
                        if origins is not None
                        else [-1] * len(target)
                    )
                error = prediction - target
                if not torch.isfinite(error).all():
                    raise ValueError("Nonfinite forecast")
                total_abs += error.abs().sum().item()
                total_sq += error.square().sum().item()
                count += error.numel()
                if arrays:
                    targets.append(target.cpu().numpy())
                    predictions.append(prediction.cpu().numpy())
    finally:
        if flow:
            model.num_samples = previous_samples
    if not count:
        raise ValueError("Empty evaluation loader")
    result = {
        "MSE": total_sq / count,
        "RMSE": math.sqrt(total_sq / count),
        "MAE": total_abs / count,
    }
    result["record_metrics"] = records.result()
    # Preserve historical pooled float32 reduction for checkpoint selection.
    result["record_metrics"]["pooled"].update(
        {k: result[k] for k in ["MAE", "MSE", "RMSE"]}
    )
    if prefixes:
        result["prefix_metrics"] = {
            str(k): acc.result() for k, acc in prefix_metrics.items()
        }
    if prefixes and target.shape[1] in prefixes:
        result["prefix_metrics"][str(target.shape[1])]["pooled"].update(
            {k: result[k] for k in ["MAE", "MSE", "RMSE"]}
        )
    if arrays:
        result.update(
            targets=np.concatenate(targets),
            predictions=np.concatenate(predictions),
            record_ids=np.asarray(ids_saved),
            forecast_origins=np.asarray(origins_saved, dtype=np.int64),
        )
    return result


def write_history(path, rows, fields):
    if any(set(r) != set(fields) for r in rows):
        raise ValueError("Inconsistent history fields")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@dataclass
class StabilizationConfig:
    min_delta: float = 0.001
    patience: int = 10
    window: int = 10
    min_epochs: int = 20
    rmse_relative_range: float = 0.02
    loss_relative_trend: float = 0.02
    gap_relative_increase: float = 0.05
    stop_policy: str = "stabilization"


class StabilizationAssessment:
    def __init__(self, config):
        self.config = config
        if config.patience < 1 or config.window < 2 or config.min_epochs < 1:
            raise ValueError("Invalid patience/window/min_epochs")
        if any(
            not math.isfinite(v) or v < 0
            for v in [
                config.min_delta,
                config.rmse_relative_range,
                config.loss_relative_trend,
                config.gap_relative_increase,
            ]
        ):
            raise ValueError("Thresholds must be finite and nonnegative")
        if config.stop_policy not in ["stabilization", "patience", "none"]:
            raise ValueError("Invalid stop policy")
        self.anchor = float("inf")
        self.stale = 0
        self.records = []

    @staticmethod
    def relative_trend(values):
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        center = (len(values) - 1) / 2
        slope = sum((i - center) * v for i, v in enumerate(values)) / sum(
            (i - center) ** 2 for i in range(len(values))
        )
        return abs(slope * (len(values) - 1)) / mean

    def update(self, rmse, train_loss, validation_loss):
        c = self.config
        if any(
            not math.isfinite(x) or x < 0 for x in [rmse, train_loss, validation_loss]
        ):
            raise ValueError("Loss/metric must be finite and nonnegative")
        if rmse < self.anchor - c.min_delta:
            self.anchor = rmse
            self.stale = 0
        else:
            self.stale += 1
        self.records.append((rmse, train_loss, validation_loss))
        n = len(self.records)
        enough = n >= max(c.min_epochs, 2 * c.window)
        stable = False
        diagnostics = {}
        if enough:
            late = self.records[-c.window :]
            previous = self.records[-2 * c.window : -c.window]
            metrics = [r[0] for r in late]
            mean = sum(metrics) / len(metrics)
            spread = (max(metrics) - min(metrics)) / mean if mean else 0.0
            tr = self.relative_trend([r[1] for r in late])
            va = self.relative_trend([r[2] for r in late])

            def gap(block):
                train = sum(r[1] for r in block) / len(block)
                val = sum(r[2] for r in block) / len(block)
                return val / max(train, 1e-12)

            old_gap, new_gap = gap(previous), gap(late)
            increase = max(0.0, (new_gap - old_gap) / max(abs(old_gap), 1e-12))
            diagnostics = {
                "rmse_relative_range": spread,
                "training_loss_relative_trend": tr,
                "validation_loss_relative_trend": va,
                "gap_relative_increase": increase,
            }
            stable = (
                spread <= c.rmse_relative_range
                and tr <= c.loss_relative_trend
                and va <= c.loss_relative_trend
                and increase <= c.gap_relative_increase
            )
        patience_met = self.stale >= c.patience and enough
        practical = patience_met and stable
        status = (
            "practical_stabilization"
            if practical
            else "patience_exhausted_without_stabilization"
            if patience_met
            else "insufficient_evidence"
        )
        stop = (practical and c.stop_policy == "stabilization") or (
            patience_met and c.stop_policy == "patience"
        )
        return {
            "status": status,
            "stop": stop,
            "patience_counter": self.stale,
            "mathematical_convergence_demonstrated": False,
            **diagnostics,
        }

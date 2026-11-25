"""Bounded synthetic tests. Never opens CWRU signals or runs a real experiment."""

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from convergence_utils import *
from run_convergence_experiment import parser, fit
from plot_convergence_losses import read_series
from sequence_data import ManifestSequenceDataset
from tsflow_data import TSFlowManifestDataset

torch.set_num_threads(1)


class ToyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.num_samples = 8

    def _extract_features(self, b):
        x1 = torch.cat([b["past_target"], b["future_target"]], dim=1).unsqueeze(-1)
        x0 = torch.cat(
            [b["past_target"], torch.randn_like(b["future_target"])], dim=1
        ).unsqueeze(-1)
        return x1, x0, None, None, None, b["past_target"]

    def p_losses(self, x1, x0, t, features):
        return ((x1 - x0) - self.weight * t.unsqueeze(-1)).square().mean()

    def forward(self, past, observed, mean):
        return (past[:, -4:] * self.weight).unsqueeze(1).repeat(1, self.num_samples, 1)


class SyntheticTests(unittest.TestCase):
    def test_stabilization_sequences(self):
        c = StabilizationConfig(patience=3, window=3, min_epochs=6)
        m = StabilizationAssessment(c)
        for _ in range(6):
            r = m.update(1.0, 1.0, 1.0)
        self.assertTrue(r["stop"])
        self.assertEqual(r["status"], "practical_stabilization")
        self.assertFalse(r["mathematical_convergence_demonstrated"])
        m = StabilizationAssessment(c)
        for i in range(12):
            r = m.update(1.0 - 0.02 * i, 1.0, 1.0)
        self.assertFalse(r["stop"])
        m = StabilizationAssessment(c)
        for i in range(12):
            r = m.update(1.0 + 0.1 * (i % 2), 1.0, 1.0)
        self.assertFalse(r["stop"])
        self.assertEqual(r["status"], "patience_exhausted_without_stabilization")
        m = StabilizationAssessment(c)
        for i in range(12):
            r = m.update(1.0, 1.0 - 0.02 * i, 1.0 + 0.02 * i)
        self.assertFalse(r["stop"])
        m = StabilizationAssessment(
            StabilizationConfig(
                patience=3, window=3, min_epochs=6, stop_policy="patience"
            )
        )
        for i in range(12):
            r = m.update(1.0 + 0.1 * (i % 2), 1.0, 1.0)
        self.assertTrue(r["stop"])
        self.assertNotEqual(r["status"], "practical_stabilization")
        with self.assertRaises(ValueError):
            m.update(float("nan"), 1.0, 1.0)
        with self.assertRaises(ValueError):
            StabilizationAssessment(StabilizationConfig(window=1))

    def test_validation_rng_mode_and_parameters(self):
        model = ToyFlow()
        model.train()
        model.weight.grad = torch.tensor(3.0)
        batch = {
            "past_target": torch.ones(3, 8),
            "future_target": torch.full((3, 4), 2.0),
            "past_observed_values": torch.ones(3, 8),
            "mean": torch.zeros(3, 1),
        }
        loader = DataLoader(
            [{k: v[i] for k, v in batch.items()} for i in range(3)], batch_size=2
        )
        before = model.weight.detach().clone()
        grad = model.weight.grad.clone()
        state = torch.get_rng_state().clone()
        py = random.getstate()
        ns = np.random.get_state()
        value = validation_flow_loss(model, loader, "cpu", 123, 2)
        self.assertEqual(value, validation_flow_loss(model, loader, "cpu", 123, 2))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertTrue(torch.equal(grad, model.weight.grad))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(ns[1], np.random.get_state()[1]))
        with evaluation_context(model, "cpu", 123):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            with self.assertRaises(ValueError):
                with evaluation_context(model, "cpu", 123):
                    raise ValueError("test")
        self.assertTrue(model.training)
        forecast = validation_forecast(model, loader, "cpu", 123, flow=True)
        self.assertAlmostEqual(forecast["MSE"], 3.61, places=5)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_weighted_mse_and_log_validation(self):
        model = torch.nn.Identity()
        loader = DataLoader(
            TensorDataset(torch.tensor([[1.0], [1.0], [3.0]]), torch.zeros(3, 1)),
            batch_size=2,
        )
        v = validation_forecast(model, loader, "cpu", 1)
        self.assertAlmostEqual(v["MSE"], 11 / 3)
        self.assertAlmostEqual(v["RMSE"] ** 2, v["MSE"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.csv"
            p.write_text(
                "epoch,training_forecast_MSE,validation_forecast_MSE\n1,1,2\n2,0,2\n"
            )
            with self.assertRaises(ValueError):
                read_series(p, "timesnet")
            p.write_text(
                "epoch,training_forecast_MSE,validation_forecast_MSE\n1,1,2\n2,1,2\n"
            )
            self.assertEqual(read_series(p, "timesnet")[0], [1, 2])
            with self.assertRaises(ValueError):
                write_history(p, [{"epoch": 1}], TIMESNET_FIELDS)

    def test_synthetic_split_and_two_epoch_csv(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "synthetic"
            data.mkdir()
            with (data / "manifest.csv").open("w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["record_id", "split", "relative_path"]
                )
                w.writeheader()
                for s, offset in [("train", 0), ("validation", 100), ("test", 200)]:
                    np.save(
                        data / (s + ".npy"), np.arange(20, dtype="float32") + offset
                    )
                    w.writerow(
                        {"record_id": s, "split": s, "relative_path": s + ".npy"}
                    )
            t = ManifestSequenceDataset(data, "train", 8, 4, stride=2)
            v = ManifestSequenceDataset(data, "validation", 8, 4, stride=4)
            self.assertEqual(t.record_ids, ["train"])
            self.assertEqual(v.record_ids, ["validation"])
            self.assertEqual(len(t), 5)
            self.assertEqual(len(v), 3)
            self.assertTrue(
                torch.equal(v[0][1], torch.arange(108, 112, dtype=torch.float32))
            )
            for name in ["timesnet", "tsflow"]:
                args = parser().parse_args(
                    [
                        "--model",
                        name,
                        "--max-epochs",
                        "2",
                        "--output-dir",
                        str(root / name),
                        "--device",
                        "cpu",
                    ]
                )
                out = root / name
                out.mkdir()
                if name == "timesnet":
                    loaders = {
                        "train": DataLoader(t, batch_size=2),
                        "validation": DataLoader(v, batch_size=2),
                    }
                    model = torch.nn.Linear(8, 4)
                else:
                    train = TSFlowManifestDataset(data, "train", 8, 4, 2)
                    val = TSFlowManifestDataset(data, "validation", 8, 4, 4)
                    loaders = {
                        "train": DataLoader(train, batch_size=2),
                        "validation": DataLoader(val, batch_size=2),
                    }
                    model = ToyFlow()
                result, history = fit(model, loaders, args, out)
                self.assertEqual(result["optimizer_steps"], 6)
                self.assertEqual(result["termination_reason"], "maximum_epoch_budget")
                self.assertFalse(result["test_evaluated"])
                self.assertEqual(history[-1]["optimizer_step"], 6)
                self.assertEqual(read_series(out / "history.csv", name)[0], [1, 2])
                with (out / "history.csv").open() as f:
                    self.assertEqual(
                        list(csv.DictReader(f).fieldnames),
                        FLOW_FIELDS if name == "tsflow" else TIMESNET_FIELDS,
                    )

    def test_real_model_synthetic_objectives(self):
        from run_flow_experiment import create_model
        from run_experiments import TimesNetLightning

        torch.manual_seed(42)
        past = torch.randn(1, 512)
        target = torch.randn(1, 96)
        batch = {
            "past_target": past,
            "future_target": target,
            "past_observed_values": torch.ones_like(past),
            "future_observed_values": torch.ones_like(target),
            "mean": torch.zeros(1, 1),
        }
        batch["record_id"] = ["synthetic"]
        batch["source_forecast_start"] = torch.tensor([512])
        flow = create_model(512, 96, 0.001, 8)
        # One synthetic gradient calculation, no optimizer step or actual data.
        flow.train()
        loss, n = flow_objective(flow, batch)
        self.assertEqual(n, 608)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        before = {k: p.detach().clone() for k, p in flow.named_parameters()}
        grads = {
            k: p.grad.clone() for k, p in flow.named_parameters() if p.grad is not None
        }
        loader = DataLoader([{k: v[0] for k, v in batch.items()}], batch_size=1)
        value = validation_flow_loss(flow, loader, "cpu", 77, 1)
        self.assertTrue(np.isfinite(value))
        for k, p in flow.named_parameters():
            self.assertTrue(torch.equal(before[k], p))
            if k in grads:
                self.assertTrue(torch.equal(grads[k], p.grad))
        times = TimesNetLightning(input_length=512, prediction_length=96)
        times.train()
        loss = (times(past) - target).square().mean()
        loss.backward()
        before = {k: p.detach().clone() for k, p in times.named_parameters()}
        metrics = validation_forecast(
            times, DataLoader(TensorDataset(past, target), batch_size=1), "cpu", 77
        )
        self.assertAlmostEqual(metrics["MSE"], metrics["RMSE"] ** 2)
        for k, p in times.named_parameters():
            self.assertTrue(torch.equal(before[k], p))
        self.assertTrue(times.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)

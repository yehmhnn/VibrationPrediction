"Synthetic metrics/preprocessing tests and read-only approved-data integrity checks."

import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat, loadmat
from torch.utils.data import DataLoader, TensorDataset, Dataset
from cwru_protocols import PROTOCOL_B, PROTOCOL_DATASETS
from preprocess_cwru_protocol_b import preprocess_assignments
from validate_protocol_b import verify_protocol_b
from run_convergence_experiment import parser, fit, verify_primary
from record_evaluation import RecordMetrics, IdentifiedDataset
from convergence_utils import validation_forecast
from sequence_data import ManifestSequenceDataset
from tsflow_data import TSFlowManifestDataset

torch.set_num_threads(1)


class NeverRead(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):
        raise AssertionError("Test/reference must not be read during fit")


class ProtocolTests(unittest.TestCase):
    def synthetic_sources(self, root):
        specs = [dict(PROTOCOL_B[i]) for i in [0, 1, 2, 3]]
        for i, s in enumerate(specs):
            s["num_samples"] = 650 + i * 32
            p = root / s["source_relative_path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            x = np.arange(s["num_samples"], dtype=float) / 100 + 10 * i
            savemat(
                p,
                {
                    s["source_key"]: x.reshape(-1, 1),
                    "X" + str(i) + "RPM": np.array([[s["rpm"]]]),
                },
            )
        return specs

    def test_preprocessing_train_only_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            specs = self.synthetic_sources(root / "raw")
            rows, n = preprocess_assignments(root / "raw", root / "out1", specs)
            train = np.concatenate(
                [
                    loadmat(root / "raw" / s["source_relative_path"])[
                        s["source_key"]
                    ].ravel()
                    for s in specs
                    if s["split"] == "train"
                ]
            )
            self.assertEqual(n["train_mean"], float(train.mean()))
            self.assertEqual(n["train_std"], float(train.std()))
            self.assertFalse(n["uses_validation_or_test"])
            preprocess_assignments(root / "raw", root / "out2", specs)
            for p in (root / "out1").iterdir():
                self.assertEqual(p.read_bytes(), (root / "out2" / p.name).read_bytes())
            for row in rows:
                raw = loadmat(row["source_path"])[row["source_key"]].ravel()
                expected = ((raw - n["train_mean"]) / n["train_std"]).astype("float32")
                self.assertTrue(
                    np.array_equal(
                        np.load(root / "out1" / row["relative_path"]), expected
                    )
                )
            with self.assertRaises(FileExistsError):
                preprocess_assignments(root / "raw", root / "out1", specs)
            with self.assertRaises(ValueError):
                preprocess_assignments(root / "raw", root / "dup", specs + [specs[0]])
            # Changing only held-out amplitudes must not change training statistics.
            for i, s in enumerate(specs[2:]):
                savemat(
                    root / "raw" / s["source_relative_path"],
                    {
                        s["source_key"]: np.full((s["num_samples"], 1), 10000.0),
                        "XrpmRPM": np.array([[s["rpm"]]]),
                    },
                )
            _, n2 = preprocess_assignments(root / "raw", root / "out3", specs)
            self.assertEqual(n["train_mean"], n2["train_mean"])
            self.assertEqual(n["train_std"], n2["train_std"])

    def test_reject_bad_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            specs = self.synthetic_sources(root / "raw")
            s = specs[0]
            p = root / "raw" / s["source_relative_path"]
            savemat(
                p,
                {
                    "Wrong_DE_time": np.ones((s["num_samples"], 1)),
                    "RPM": np.array([[s["rpm"]]]),
                },
            )
            with self.assertRaises(ValueError):
                preprocess_assignments(root / "raw", root / "bad1", specs)
            x = np.ones((s["num_samples"], 1))
            x[0] = np.nan
            savemat(p, {s["source_key"]: x, "RPM": np.array([[s["rpm"]]])})
            with self.assertRaises(ValueError):
                preprocess_assignments(root / "raw", root / "bad2", specs)
            self.assertFalse((root / "bad2").exists())

    def test_unequal_record_metrics(self):
        m = RecordMetrics(
            {
                "a": {"condition": "inner_race_fault", "load_id": "2", "rpm": "1748"},
                "b": {"condition": "ball_fault", "load_id": "2", "rpm": "1748"},
            }
        )
        m.update(
            torch.tensor([[1.0, 1.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]),
            torch.zeros(4, 2),
            ["a", "b", "b", "b"],
        )
        r = m.result()
        self.assertEqual(r["pooled"]["MAE"], 2.5)
        self.assertAlmostEqual(r["pooled"]["RMSE"], np.sqrt(7))
        self.assertEqual(r["equal_record"]["MAE"], 2.0)
        self.assertEqual(r["equal_record"]["RMSE"], 2.0)
        self.assertEqual(r["per_record"]["b"]["num_windows"], 3)
        self.assertEqual(r["per_record"]["b"]["num_target_elements"], 6)
        self.assertEqual(r["per_fault_type"]["ball_fault"]["RMSE"], 3.0)
        self.assertEqual(r["per_record"]["a"]["load"], 2)

    def test_prefixes_ids_and_zero_test_access(self):
        x = torch.zeros(4, 512)
        x[0, -96:] = 1.0
        x[1:, -96:] = 3.0
        y = torch.zeros(4, 96)

        class Echo(torch.nn.Module):
            def forward(self, x):
                return x[:, -96:]

        class Data(Dataset):
            record_metadata = {
                "a": {"condition": "inner_race_fault"},
                "b": {"condition": "ball_fault"},
            }

            def __len__(self):
                return 4

            def __getitem__(self, i):
                return x[i], y[i], "a" if i == 0 else "b", 100 + i * 96

        v = validation_forecast(
            Echo(),
            DataLoader(Data(), batch_size=2),
            "cpu",
            42,
            arrays=True,
            prefixes=(24, 48, 72, 96),
        )
        self.assertEqual(v["record_ids"].tolist(), ["a", "b", "b", "b"])
        self.assertEqual(v["forecast_origins"].tolist(), [100, 196, 292, 388])
        self.assertEqual(v["prefix_metrics"]["96"]["pooled"]["RMSE"], v["RMSE"])
        for k in [24, 48, 72, 96]:
            self.assertEqual(
                v["prefix_metrics"][str(k)]["pooled"]["num_target_elements"], 4 * k
            )
        with tempfile.TemporaryDirectory() as d:
            args = parser().parse_args(
                [
                    "--model",
                    "timesnet",
                    "--protocol",
                    "multi_record_B",
                    "--max-epochs",
                    "2",
                    "--output-dir",
                    d,
                    "--device",
                    "cpu",
                ]
            )
            train = DataLoader(
                TensorDataset(torch.ones(3, 8), torch.zeros(3, 4)), batch_size=2
            )
            val = DataLoader(
                TensorDataset(torch.ones(2, 8), torch.zeros(2, 4)), batch_size=2
            )
            result, rows = fit(
                torch.nn.Linear(8, 4),
                {
                    "train": train,
                    "validation": val,
                    "test": DataLoader(NeverRead()),
                    "in_condition_reference": DataLoader(NeverRead()),
                },
                args,
                Path(d),
            )
            self.assertFalse(result["test_evaluated"])
            self.assertFalse(result["in_condition_reference_evaluated"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                result["validation_seconds"], sum(r["validation_seconds"] for r in rows)
            )
            self.assertGreaterEqual(
                result["total_elapsed_seconds"], result["training_seconds"]
            )
            self.assertEqual(
                len((Path(d) / "validation_records.jsonl").read_text().splitlines()), 2
            )
            self.assertEqual(
                len((Path(d) / "assessment_history.jsonl").read_text().splitlines()), 2
            )

    def test_approved_data_and_single_backward_compatibility(self):
        verify_primary(PROTOCOL_DATASETS["single_105_0"])
        meta, manifest = verify_protocol_b(PROTOCOL_DATASETS["multi_record_B"])
        self.assertEqual(
            meta["split_record_ids"]["train"],
            ["105_0", "106_1", "118_0", "119_1", "130@6_0", "131@6_1"],
        )
        for protocol, counts in [
            ("single_105_0", {"train": 2210, "validation": 242, "test": 242}),
            ("multi_record_B", {"train": 22754, "validation": 3785, "test": 3806}),
        ]:
            for split, n in counts.items():
                root = PROTOCOL_DATASETS[protocol]
                stride = 32 if split == "train" else 96
                tn = ManifestSequenceDataset(root, split, 512, 96, stride)
                flow = TSFlowManifestDataset(root, split, 512, 96, stride)
                self.assertEqual(len(tn), n)
                self.assertEqual(tn.window_index, flow.windows.window_index)
                it = IdentifiedDataset(tn)
                iff = IdentifiedDataset(flow, flow=True)
                for index in [0, n // 2, n - 1]:
                    a, b = it[index], iff[index]
                    self.assertTrue(torch.equal(a[0], b["past_target"]))
                    self.assertTrue(torch.equal(a[1], b["future_target"]))
                    self.assertEqual(a[2], b["record_id"])
                    self.assertEqual(a[3], b["source_forecast_start"])
                    self.assertEqual(a[0].shape, (512,))
                    self.assertEqual(a[1].shape, (96,))
                for ri, start in tn.window_index:
                    self.assertLessEqual(start + 608, tn.records[ri][1].size)
                fb = next(iter(DataLoader(iff, batch_size=2)))
                tb = next(iter(DataLoader(it, batch_size=2)))
                self.assertEqual(fb["past_target"].shape, (2, 512))
                self.assertEqual(fb["future_target"].shape, (2, 96))
                self.assertTrue(torch.equal(fb["future_target"], tb[1]))
        # Independently validate processed values against raw signals and train-only statistics.
        r = PROTOCOL_DATASETS["multi_record_B"]
        norm = json.loads((r / "normalization.json").read_text())
        all_train = []
        for row in manifest:
            raw = np.asarray(
                loadmat(row["source_path"])[row["source_key"]], dtype=float
            ).ravel()
            expected = ((raw - norm["train_mean"]) / norm["train_std"]).astype(
                "float32"
            )
            self.assertTrue(np.array_equal(np.load(r / row["relative_path"]), expected))
            if row["split"] == "train":
                all_train.append(raw)
        raw = np.concatenate(all_train)
        self.assertEqual(float(raw.mean()), norm["train_mean"])
        self.assertEqual(float(raw.std()), norm["train_std"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

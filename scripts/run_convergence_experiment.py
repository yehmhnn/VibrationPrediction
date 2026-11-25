"""Fresh primary convergence run. Requires an explicit budget and unused output path."""

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from cwru_protocols import PROTOCOL_DATASETS
from record_evaluation import IdentifiedDataset
from convergence_utils import (
    FLOW_FIELDS,
    TIMESNET_FIELDS,
    StabilizationConfig,
    StabilizationAssessment,
    flow_objective,
    validation_flow_loss,
    validation_forecast,
    move,
    write_history,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        choices=["single_105_0", "multi_record_B"],
        default="single_105_0",
        help="Recorded explicitly in config; legacy default is single105_0",
    )
    p.add_argument("--model", choices=["tsflow", "timesnet"], required=True)
    p.add_argument(
        "--max-epochs",
        type=int,
        required=True,
        help="Budget must be approved separately; no default",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must not exist; no resume/overwrite",
    )
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--fm-evaluation-draws", type=int, default=4)
    p.add_argument("--evaluation-seed", type=int, default=420001)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--stability-window", type=int, default=10)
    p.add_argument("--min-assessment-epochs", type=int, default=20)
    p.add_argument("--rmse-relative-range", type=float, default=0.02)
    p.add_argument("--loss-relative-trend", type=float, default=0.02)
    p.add_argument("--gap-relative-increase", type=float, default=0.05)
    p.add_argument(
        "--stop-policy",
        choices=["stabilization", "patience", "none"],
        default="stabilization",
    )
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Optional once after restoring selected checkpoint",
    )
    p.add_argument(
        "--in-condition-reference",
        action="store_true",
        help=(
            "Optional post-selection evaluation of training-condition records, never "
            "independent test evidence"
        ),
    )
    return p


def assessment_config(a):
    return StabilizationConfig(
        a.min_delta,
        a.patience,
        a.stability_window,
        a.min_assessment_epochs,
        a.rmse_relative_range,
        a.loss_relative_trend,
        a.gap_relative_increase,
        a.stop_policy,
    )


def verify_primary(dataset_dir):
    meta = json.loads((dataset_dir / "metadata.json").read_text())
    expected = {
        "train": [0, 71319],
        "validation": [72519, 96292],
        "test": [97492, 121265],
    }
    if (
        meta.get("parent_record_id") != "105_0"
        or meta.get("sampling_rate_hz") != 12000
        or meta.get("boundaries") != expected
        or meta.get("gap_samples") != 1200
    ):
        raise ValueError("Not the primary105_0 chronological protocol")
    if meta.get("normalization") != "z-score using chronological training region only":
        raise ValueError("Unexpected normalization")
    with (dataset_dir / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 3:
        raise ValueError("Expected exactly three primary regions")
    for split, bounds in expected.items():
        matches = [r for r in rows if r["split"] == split]
        if len(matches) != 1:
            raise ValueError("Unexpected split assignment")
        r = matches[0]
        if (
            r["parent_record_id"] != "105_0"
            or r["record_id"] != "105_0_" + split
            or [int(r["source_start"]), int(r["source_end_exclusive"])] != bounds
            or int(r["sampling_rate_hz"]) != 12000
            or r["source_key"] != "X105_DE_time"
        ):
            raise ValueError("Manifest protocol mismatch")
        rel = Path(r["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path")
        signal = np.load(dataset_dir / rel, mmap_mode="r")
        if (
            signal.ndim != 1
            or signal.size != bounds[1] - bounds[0]
            or int(r["num_samples"]) != signal.size
        ):
            raise ValueError("Signal length mismatch")
    return meta, rows


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def fit(model, loaders, args, output):
    """Caller supplies fresh directory; synthetic tests supply synthetic loaders."""
    if any(
        (output / name).exists()
        for name in [
            "history.csv",
            "best.pt",
            "result.json",
            "validation_records.jsonl",
            "assessment_history.jsonl",
        ]
    ):
        raise FileExistsError("Refusing to overwrite existing experiment artifacts")
    is_flow = args.model == "tsflow"
    fields = FLOW_FIELDS if is_flow else TIMESNET_FIELDS
    train_key = fields[2]
    val_key = fields[3]
    monitor = StabilizationAssessment(assessment_config(args))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rows = []
    steps = 0
    best = float("inf")
    best_epoch = None
    optimization_seconds = 0.0
    sync(args.device)
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        sync(args.device)
        tick = time.perf_counter()
        for batch in loaders["train"]:
            batch = move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            if is_flow:
                loss, n = flow_objective(model, batch)
            else:
                x, y = batch[:2]
                error = model(x.float()) - y.float()
                loss = error.square().mean()
                n = y.numel()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            if is_flow:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            steps += 1
            total += loss.item() * n
            count += n
        if not count:
            raise ValueError("Empty training loader")
        sync(args.device)
        optimization_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        fm = (
            validation_flow_loss(
                model,
                loaders["validation"],
                args.device,
                args.evaluation_seed,
                args.fm_evaluation_draws,
            )
            if is_flow
            else None
        )
        forecasts = validation_forecast(
            model, loaders["validation"], args.device, 10042, flow=is_flow
        )
        sync(args.device)
        validation_seconds = time.perf_counter() - tick
        train_loss = total / count
        validation_loss = fm if is_flow else forecasts["MSE"]
        assessment = monitor.update(forecasts["RMSE"], train_loss, validation_loss)
        if forecasts["RMSE"] < best:
            best = forecasts["RMSE"]
            best_epoch = epoch
            best_record_metrics = forecasts["record_metrics"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimizer_step": steps,
                    "validation_forecast_RMSE": best,
                    "validation_forecast_MAE": forecasts["MAE"],
                },
                output / "best.pt",
            )
        row = {
            "epoch": epoch,
            "optimizer_step": steps,
            train_key: train_loss,
            val_key: validation_loss,
            "validation_forecast_RMSE": forecasts["RMSE"],
            "validation_forecast_MAE": forecasts["MAE"],
            "elapsed_training_seconds": time.perf_counter() - started,
            "optimization_seconds": optimization_seconds,
            "validation_seconds": validation_seconds,
            "assessment": assessment["status"],
        }
        rows.append(row)
        write_history(output / "history.csv", rows, fields)
        with (output / "validation_records.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, **forecasts["record_metrics"]}) + "\n")
        with (output / "assessment_history.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, **assessment}) + "\n")
        print(
            f"epoch={epoch} step={steps} train={train_loss:.6f} "
            f"validation_objective={validation_loss:.6f} "
            f"validation_RMSE={forecasts['RMSE']:.6f} "
            f"assessment={assessment['status']}",
            flush=True,
        )
        if assessment["stop"]:
            break
    result = {
        "epochs_completed": len(rows),
        "optimizer_steps": steps,
        "best_epoch": best_epoch,
        "best_validation_forecast_RMSE": best,
        "termination_reason": assessment["status"]
        if assessment["stop"]
        else "maximum_epoch_budget",
        "assessment": assessment,
        "criterion": asdict(monitor.config),
        "training_seconds": time.perf_counter() - started,
        "optimization_seconds": optimization_seconds,
        "test_evaluated": False,
        "in_condition_reference_evaluated": False,
        "best_validation_record_metrics": None,
    }
    result["validation_seconds"] = sum(r["validation_seconds"] for r in rows)
    result["best_validation_record_metrics"] = best_record_metrics
    if args.evaluate_test or getattr(args, "in_condition_reference", False):
        checkpoint = torch.load(
            output / "best.pt", map_location=args.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    if args.evaluate_test:
        test = validation_forecast(
            model,
            loaders["test"],
            args.device,
            20042,
            flow=is_flow,
            arrays=True,
            prefixes=(24, 48, 72, 96),
        )
        np.save(output / "record_ids.npy", test.pop("record_ids"))
        np.save(output / "forecast_origins.npy", test.pop("forecast_origins"))
        np.save(output / "targets.npy", test.pop("targets"))
        np.save(output / "predictive_mean.npy", test.pop("predictions"))
        result.update(test_evaluated=True, test_metrics=test)
    if getattr(args, "in_condition_reference", False):
        reference = validation_forecast(
            model,
            loaders["in_condition_reference"],
            args.device,
            30042,
            flow=is_flow,
            prefixes=(24, 48, 72, 96),
        )
        result.update(
            in_condition_reference_evaluated=True,
            in_condition_reference={
                "label": (
                    "selected checkpoint on training-condition records; reused training"
                    " data; not independent evidence; excluded from selection/stopping"
                ),
                **reference,
            },
        )
    result["total_elapsed_seconds"] = time.perf_counter() - started
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, rows


def main():
    p = parser()
    args = p.parse_args()
    if args.max_epochs < 1 or args.fm_evaluation_draws < 1:
        p.error("max-epochs and fm-evaluation-draws must be positive")
    try:
        StabilizationAssessment(assessment_config(args))
    except ValueError as e:
        p.error(str(e))
    if args.output_dir.exists():
        p.error("Output directory already exists; refusing overwrite/resume")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    args.dataset_dir = args.dataset_dir or PROTOCOL_DATASETS[args.protocol]
    if args.protocol == "single_105_0":
        meta, manifest = verify_primary(args.dataset_dir)
    else:
        from validate_protocol_b import verify_protocol_b

        meta, manifest = verify_protocol_b(args.dataset_dir)
    from run_flow_experiment import seed_everything

    seed_everything(42)
    torch.use_deterministic_algorithms(args.model == "timesnet")
    if args.model == "tsflow":
        from run_flow_experiment import create_model
        from tsflow_data import TSFlowManifestDataset

        datasets = {
            s: TSFlowManifestDataset(
                args.dataset_dir,
                split=s,
                context_length=512,
                prediction_length=96,
                stride=32 if s == "train" else 96,
            )
            for s in ["train", "validation", "test"]
        }
        model = create_model(512, 96, 0.001, 8)
    else:
        from run_experiments import TimesNetLightning
        from sequence_data import ManifestSequenceDataset

        datasets = {
            s: ManifestSequenceDataset(
                args.dataset_dir, s, 512, 96, stride=32 if s == "train" else 96
            )
            for s in ["train", "validation", "test"]
        }
        model = TimesNetLightning(input_length=512, prediction_length=96)
    if args.in_condition_reference:
        datasets["in_condition_reference"] = (
            TSFlowManifestDataset(
                args.dataset_dir,
                split="train",
                context_length=512,
                prediction_length=96,
                stride=96,
            )
            if args.model == "tsflow"
            else ManifestSequenceDataset(args.dataset_dir, "train", 512, 96, stride=96)
        )
    datasets = {
        s: IdentifiedDataset(d, flow=args.model == "tsflow")
        for s, d in datasets.items()
    }
    model = model.to(args.device)
    generator = torch.Generator().manual_seed(42)
    loaders = {
        s: DataLoader(
            d,
            batch_size=16 if args.model == "tsflow" and s != "train" else 32,
            shuffle=s == "train",
            num_workers=0,
            generator=generator if s == "train" else None,
        )
        for s, d in datasets.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        args.dataset_dir / "metadata.json",
        args.dataset_dir / "manifest.csv",
    ] + [args.dataset_dir / r["relative_path"] for r in manifest]
    code = Path(__file__).resolve().parent
    sources += [
        code / "run_convergence_experiment.py",
        code / "convergence_utils.py",
        code / "record_evaluation.py",
        code / "cwru_protocols.py",
        code / "sequence_data.py",
        code / "tsflow_data.py",
        code / "run_flow_experiment.py"
        if args.model == "tsflow"
        else code / "run_experiments.py",
    ]
    configuration = {
        **vars(args),
        "L": 512,
        "H": 96,
        "seed": 42,
        "train_stride": 32,
        "eval_stride": 96,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_origins": {s: len(d) for s, d in datasets.items()},
        "split_record_ids": {s: d.record_ids for s, d in datasets.items()},
        "normalization": meta["normalization"],
        "selection_metric": "lowest validation_forecast_RMSE, regardless of min_delta",
        "criterion": asdict(assessment_config(args)),
        "training_loss_measurement": (
            "element-weighted batch losses during optimization in training mode; not a "
            "post-epoch evaluation"
        ),
        "validation_fm_protocol": (
            "fixed seeds evaluation_seed+draw, fixed order/batch size; same "
            "context+future p_losses as training; isolated RNG; evaluation mode; no "
            "gradients"
        ),
        "forecast_evaluation_seed": 10042,
        "num_samples": 8 if args.model == "tsflow" else None,
        "sampling_steps": 8 if args.model == "tsflow" else None,
        "timing": (
            "elapsed includes optimization+validation+logging; optimization_seconds "
            "excludes validation, init, and final test"
        ),
        "deterministic_algorithms": args.model == "timesnet",
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "source_sha256": {
            str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in sources
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, default=str) + "\n"
    )
    fit(model, loaders, args, args.output_dir)


if __name__ == "__main__":
    main()

"""Train and evaluate one record-aware conditional Flow Matching experiment.

This intentionally small driver uses the same manifest windows as the TimesNet
experiments. Model selection is based only on predictive-mean validation RMSE;
the test split is evaluated once after restoring the selected checkpoint.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tsflow.model import TSFlowCond

from tsflow_data import TSFlowManifestDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def create_model(lookback, horizon, learning_rate, sampling_steps):
    return TSFlowCond(
        setting="univariate",
        target_dim=1,
        context_length=lookback,
        prediction_length=horizon,
        backbone_params={
            "input_dim": 1,
            "output_dim": 1,
            "step_emb": 32,
            "num_residual_blocks": 1,
            "residual_block": "s4",
            "hidden_dim": 32,
            "dropout": 0.0,
            "init_skip": False,
            "feature_skip": True,
        },
        prior_params={"kernel": "ou", "gamma": 1.0, "context_freqs": 1},
        optimizer_params={"lr": learning_rate},
        ema_params={"beta": 0.999, "update_after_step": 10, "update_every": 1},
        frequency="H",
        normalization="identity",
        use_lags=False,
        use_ema=False,
        num_steps=sampling_steps,
        solver="euler",
        matching="random",
    )


@torch.no_grad()
def evaluate(model, loader, device, num_samples, evaluation_seed, keep_samples):
    model.eval()
    model.num_samples = num_samples
    torch.manual_seed(evaluation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(evaluation_seed)

    targets = []
    predictive_means = []
    all_samples = []
    absolute_error = 0.0
    squared_error = 0.0
    value_count = 0
    started = time.perf_counter()
    for batch in loader:
        batch = to_device(batch, device)
        samples = model(
            batch["past_target"],
            batch["past_observed_values"],
            batch["mean"],
        )
        target = batch["future_target"]
        predictive_mean = samples.mean(dim=1)
        error = predictive_mean - target
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        value_count += target.numel()
        targets.append(target.cpu())
        predictive_means.append(predictive_mean.cpu())
        if keep_samples:
            all_samples.append(samples.cpu())
    runtime = time.perf_counter() - started

    target = torch.cat(targets).numpy()
    predictive_mean = torch.cat(predictive_means).numpy()
    samples = torch.cat(all_samples).numpy() if keep_samples else None
    return {
        "MAE": absolute_error / value_count,
        "RMSE": (squared_error / value_count) ** 0.5,
        "runtime_seconds": runtime,
        "target": target,
        "predictive_mean": predictive_mean,
        "samples": samples,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "datasets" / "processed" / "cwru_105_0_chronological",
    )
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stride", type=int, default=32)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.eval_stride = args.eval_stride or args.horizon

    run_id = f"{args.dataset_dir.resolve().name}_flow_L{args.lookback}_H{args.horizon}_seed{args.seed}"
    output_dir = args.output_dir or project_root / "logs" / "flow" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed)

    datasets = {
        split: TSFlowManifestDataset(
            args.dataset_dir,
            split=split,
            context_length=args.lookback,
            prediction_length=args.horizon,
            stride=args.train_stride if split == "train" else args.eval_stride,
        )
        for split in ("train", "validation", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    configuration = {
        **vars(args),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_id": run_id,
        "model": "TSFlowCond",
        "backbone": "S4, hidden_dim=32, residual_blocks=1, dropout=0",
        "prior": "OU, gamma=1.0, context_freqs=1",
        "normalization": "identity (input arrays already use training-only z-score)",
        "point_forecast": "mean over generated samples",
        "selection_metric": "validation predictive-mean RMSE",
        "split_record_ids": {split: datasets[split].record_ids for split in datasets},
        "num_origins": {split: len(datasets[split]) for split in datasets},
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "CPU",
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, default=str) + "\n", encoding="utf-8"
    )

    model = create_model(
        args.lookback, args.horizon, args.learning_rate, args.sampling_steps
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    best_validation_rmse = float("inf")
    best_epoch = None
    training_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            x1, x0, _, _, _, features = model._extract_features(batch)
            flow_time = torch.rand((x1.shape[0], 1), device=device)
            loss = model.p_losses(x1, x0, flow_time, features)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}: {loss.item()}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            loss_sum += loss.item()
            batch_count += 1

        validation = evaluate(
            model,
            loaders["validation"],
            device,
            args.num_samples,
            args.seed + 10_000,
            keep_samples=False,
        )
        train_loss = loss_sum / batch_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_MAE": validation["MAE"],
                "validation_RMSE": validation["RMSE"],
                "validation_seconds": validation["runtime_seconds"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_MAE={validation['MAE']:.6f} val_RMSE={validation['RMSE']:.6f}"
        )
        if validation["RMSE"] < best_validation_rmse:
            best_validation_rmse = validation["RMSE"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_MAE": validation["MAE"],
                    "validation_RMSE": validation["RMSE"],
                    "run_id": run_id,
                },
                checkpoint_path,
            )

    training_seconds = time.perf_counter() - training_started
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    test = evaluate(
        model,
        loaders["test"],
        device,
        args.num_samples,
        args.seed + 20_000,
        keep_samples=True,
    )
    np.save(output_dir / "targets.npy", test["target"])
    np.save(output_dir / "predictive_mean.npy", test["predictive_mean"])
    np.save(output_dir / "samples.npy", test["samples"])

    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    result = {
        "run_id": run_id,
        "dataset": args.dataset_dir.resolve().name,
        "model": "flow_matching",
        "L": args.lookback,
        "H": args.horizon,
        "seed": args.seed,
        "train_stride": args.train_stride,
        "eval_stride": args.eval_stride,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_MAE": selected["validation_MAE"],
        "best_validation_RMSE": selected["validation_RMSE"],
        "test_MAE": test["MAE"],
        "test_RMSE": test["RMSE"],
        "training_seconds": training_seconds,
        "evaluation_seconds": test["runtime_seconds"],
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "num_train_origins": len(datasets["train"]),
        "num_validation_origins": len(datasets["validation"]),
        "num_test_origins": len(datasets["test"]),
        "target_shape": list(test["target"].shape),
        "predictive_mean_shape": list(test["predictive_mean"].shape),
        "sample_shape": list(test["samples"].shape),
        "target_mean": float(test["target"].mean()),
        "target_std": float(test["target"].std()),
        "prediction_mean": float(test["predictive_mean"].mean()),
        "prediction_std": float(test["predictive_mean"].std()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "peak_gpu_memory_bytes": peak_memory,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

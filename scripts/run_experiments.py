import argparse
import csv
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from sequence_data import (
    ManifestRPMSequenceDataset,
    ManifestSequenceDataset,
    default_cwru_dataset_dir,
)
from timesnet import TimesNet


class TimesNetLightning(pl.LightningModule):
    def __init__(
        self, input_length=192, prediction_length=96, lr=1e-3, condition_rpm=False
    ):
        super().__init__()
        self.model = TimesNet(
            input_length=input_length,
            prediction_length=prediction_length,
            condition_channels=1 if condition_rpm else 0,
        )
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.condition_rpm = condition_rpm
        self.save_hyperparameters()

    def forward(self, x, condition=None):
        return self.model(x, condition)

    def unpack_batch(self, batch):
        if self.condition_rpm:
            x, rpm, y = batch
            return x, rpm, y
        x, y = batch
        return x, None, y

    def training_step(self, batch, batch_idx):
        x, rpm, y = self.unpack_batch(batch)
        pred = self(x.float(), None if rpm is None else rpm.float())
        loss = self.loss_fn(pred, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, rpm, y = self.unpack_batch(batch)
        loss = self.loss_fn(
            self(x.float(), None if rpm is None else rpm.float()), y.float()
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    mae = 0.0
    mse = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            x, condition, y = model.unpack_batch(batch)
            x = x.float().to(device)
            y = y.float().to(device)
            condition = None if condition is None else condition.float().to(device)
            pred = model(x, condition)
            mae += torch.abs(pred - y).sum().item()
            mse += ((pred - y) ** 2).sum().item()
            count += y.numel()
    return mae / count, (mse / count) ** 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for VibrationPrediction"
    )
    parser.add_argument("--output", type=str, default="logs/evaluation_results.csv")
    parser.add_argument("--dataset-dir", type=Path, default=default_cwru_dataset_dir())
    parser.add_argument("--hs", nargs="+", type=int, default=[48, 96, 192, 336])
    parser.add_argument("--l", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=16)
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=None,
        help="Validation/test stride; defaults to H for non-overlapping targets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--condition-rpm",
        action="store_true",
        help="Condition TimesNet on RPM normalized using training records only.",
    )
    args = parser.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "L",
        "H",
        "MAE",
        "RMSE",
        "seed",
        "train_stride",
        "eval_stride",
        "epochs",
        "best_val_loss",
        "training_seconds",
        "condition_rpm",
        "rpm_train_mean",
        "rpm_train_std",
        "run_id",
    ]
    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for H in args.hs:
            # Make each (seed, horizon) experiment reproducible independently of
            # the order or subset of horizons supplied on the command line.
            pl.seed_everything(args.seed, workers=True)
            L = args.l
            dataset_name = args.dataset_dir.resolve().name
            eval_stride = args.eval_stride or H
            print(f"Running CWRU experiment: L={L}, H={H}")
            rpm_mean = rpm_std = None
            dataset_class = ManifestSequenceDataset
            dataset_kwargs = {}
            if args.condition_rpm:
                with (args.dataset_dir / "manifest.csv").open(
                    newline="", encoding="utf-8"
                ) as stream:
                    train_rpms = [
                        float(row["rpm"])
                        for row in csv.DictReader(stream)
                        if row["split"] == "train"
                    ]
                if len(train_rpms) < 2:
                    raise ValueError(
                        "RPM conditioning requires at least two training records"
                    )
                rpm_mean = sum(train_rpms) / len(train_rpms)
                rpm_std = (
                    sum((value - rpm_mean) ** 2 for value in train_rpms)
                    / len(train_rpms)
                ) ** 0.5
                if rpm_std <= 0:
                    raise ValueError("Training RPM values have zero variance")
                dataset_class = ManifestRPMSequenceDataset
                dataset_kwargs = {"rpm_mean": rpm_mean, "rpm_std": rpm_std}
            datasets = {
                "train": dataset_class(
                    args.dataset_dir,
                    "train",
                    L,
                    H,
                    stride=args.train_stride,
                    **dataset_kwargs,
                ),
                "validation": dataset_class(
                    args.dataset_dir,
                    "validation",
                    L,
                    H,
                    stride=eval_stride,
                    **dataset_kwargs,
                ),
                "test": dataset_class(
                    args.dataset_dir, "test", L, H, stride=eval_stride, **dataset_kwargs
                ),
            }
            train_loader = DataLoader(
                datasets["train"],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
            validation_loader = DataLoader(
                datasets["validation"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            test_loader = DataLoader(
                datasets["test"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

            model = TimesNetLightning(
                input_length=L, prediction_length=H, condition_rpm=args.condition_rpm
            )
            model_name = "timesnet_rpm" if args.condition_rpm else "timesnet"
            checkpoint = ModelCheckpoint(
                dirpath=Path("logs")
                / "checkpoints"
                / f"{dataset_name}_{model_name}_L{L}_H{H}_seed{args.seed}",
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
            )
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                accelerator="auto",
                devices=1,
                deterministic=True,
                callbacks=[checkpoint],
                enable_progress_bar=False,
                logger=False,
                num_sanity_val_steps=0,
            )
            start_time = time.perf_counter()
            trainer.fit(model, train_loader, validation_loader)
            training_seconds = time.perf_counter() - start_time

            best_model = TimesNetLightning.load_from_checkpoint(
                checkpoint.best_model_path
            )

            mae, rmse = evaluate_model(
                best_model,
                test_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "L": L,
                    "H": H,
                    "MAE": f"{mae:.6f}",
                    "RMSE": f"{rmse:.6f}",
                    "seed": args.seed,
                    "train_stride": args.train_stride,
                    "eval_stride": eval_stride,
                    "epochs": args.epochs,
                    "best_val_loss": f"{float(checkpoint.best_model_score):.6f}",
                    "training_seconds": f"{training_seconds:.3f}",
                    "condition_rpm": args.condition_rpm,
                    "rpm_train_mean": "" if rpm_mean is None else f"{rpm_mean:.6f}",
                    "rpm_train_std": "" if rpm_std is None else f"{rpm_std:.6f}",
                    "run_id": f"{dataset_name}_{model_name}_L{L}_H{H}_seed{args.seed}",
                }
            )
            csvfile.flush()

    print(f"Saved experiment results to {out_path}")


if __name__ == "__main__":
    main()

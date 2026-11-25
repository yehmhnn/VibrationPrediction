"Run the fixed L=512, H=96 TimesNet convergence check without overwriting prior runs."

import csv
import json
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from run_experiments import TimesNetLightning, evaluate_model
from sequence_data import ManifestSequenceDataset


class ValidationHistory(Callback):
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if trainer.sanity_checking or "val_loss" not in metrics:
            return
        row = {
            "epoch": trainer.current_epoch + 1,
            "validation_MAE": float(metrics["val_mae"]),
            "validation_RMSE": float(metrics["val_loss"]) ** 0.5,
        }
        self.rows.append(row)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)


def validation_step(self, batch, batch_idx):
    x, rpm, y = self.unpack_batch(batch)
    prediction = self(x.float(), None if rpm is None else rpm.float())
    error = prediction - y.float()
    loss = error.square().mean()
    self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=y.shape[0])
    self.log(
        "val_mae",
        error.abs().mean(),
        on_step=False,
        on_epoch=True,
        batch_size=y.shape[0],
    )
    return loss


def main():
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "datasets" / "processed" / "cwru_105_0_chronological"
    output_dir = root / "logs" / "convergence" / "timesnet_L512_H96_seed42_e30"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = ValidationHistory(output_dir / "history.csv")

    seed, lookback, horizon = 42, 512, 96
    pl.seed_everything(seed, workers=True)
    train = ManifestSequenceDataset(dataset_dir, "train", lookback, horizon, stride=32)
    validation = ManifestSequenceDataset(
        dataset_dir, "validation", lookback, horizon, stride=96
    )
    test = ManifestSequenceDataset(dataset_dir, "test", lookback, horizon, stride=96)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation, batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)

    model = TimesNetLightning(input_length=lookback, prediction_length=horizon)
    model.validation_step = validation_step.__get__(model, TimesNetLightning)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        deterministic=True,
        callbacks=[checkpoint, history],
        enable_progress_bar=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    started = time.perf_counter()
    trainer.fit(model, train_loader, validation_loader)
    training_seconds = time.perf_counter() - started

    best_model = TimesNetLightning.load_from_checkpoint(checkpoint.best_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_mae, test_rmse = evaluate_model(best_model, test_loader, device=device)
    best_row = min(history.rows, key=lambda row: row["validation_RMSE"])
    result = {
        "dataset": dataset_dir.name,
        "model": "timesnet",
        "L": lookback,
        "H": horizon,
        "seed": seed,
        "train_stride": 32,
        "eval_stride": 96,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "best_epoch": best_row["epoch"],
        "best_validation_MAE": best_row["validation_MAE"],
        "best_validation_RMSE": best_row["validation_RMSE"],
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
        "training_seconds": training_seconds,
        "checkpoint_path": checkpoint.best_model_path,
        "num_train_origins": len(train),
        "num_validation_origins": len(validation),
        "num_test_origins": len(test),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

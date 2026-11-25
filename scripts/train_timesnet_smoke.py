import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from sequence_data import ManifestSequenceDataset, default_cwru_dataset_dir
from timesnet import TimesNet


class TimesNetLightning(pl.LightningModule):
    def __init__(self, input_length=192, prediction_length=96, lr=1e-3):
        super().__init__()
        self.model = TimesNet(
            input_length=input_length, prediction_length=prediction_length
        )
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.float())
        loss = self.loss_fn(pred, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x.float()), y.float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    train_data = ManifestSequenceDataset(
        default_cwru_dataset_dir(), "train", 192, 96, stride=16
    )
    validation_data = ManifestSequenceDataset(
        default_cwru_dataset_dir(), "validation", 192, 96, stride=16
    )
    # A smoke test checks plumbing, not convergence, so keep it intentionally small.
    train_data = Subset(train_data, range(min(1024, len(train_data))))
    validation_data = Subset(validation_data, range(min(256, len(validation_data))))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_data, batch_size=32, shuffle=False, num_workers=0
    )

    model = TimesNetLightning(input_length=192, prediction_length=96)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        deterministic=True,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_loader, validation_loader)
    print("Smoke training completed.")


if __name__ == "__main__":
    main()

from typing import Union, Optional, Callable, Any

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer


class BasicRegressor(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()
        self.lr = lr


    def log_descaled_values(self):
        return self.outputs, self.labels

    def forward(self, x, labels = None):
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train|MSE", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("val|MSE", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)

        self.log("test|MSE", loss, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        raise NotImplementedError



    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=False)
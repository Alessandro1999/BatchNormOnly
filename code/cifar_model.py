'''
This file contains the code regarding the pytorch lightning model
of the test on CIFAR-10 with a MLP
'''
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import wandb
from typing import *


class MLPModule(pl.LightningModule):
    '''
    A MLP with batch normalization.
    - input_dim, the input dimension;
    - hidden1_dim, dimension of the first hidden layer;
    - hidden2_dim, dimension of the second hidden layer;
    - hidden3_dim, dimension of the third hidden layer;
    - n_classes, the output dimension;
    - batch_norm_only, if true, only the batch normalization parameters are used for training.
    '''

    def __init__(self, input_dim: int, hidden1_dim: int, hidden2_dim: int, hidden3_dim: int, n_classes: int, batch_norm_only: bool = False) -> None:
        super().__init__()

        # Network structure
        self.layer_1 = nn.Linear(input_dim, hidden1_dim)
        self.batch_n1 = nn.BatchNorm1d(hidden1_dim)
        self.layer_2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.batch_n2 = nn.BatchNorm1d(hidden2_dim)
        self.layer_3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.batch_n3 = nn.BatchNorm1d(hidden3_dim)
        self.layer_output = nn.Linear(hidden3_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

        # freeze the weights of the linear layers at random inizialization (if required)
        self.layer_1.requires_grad_(not(batch_norm_only))
        self.layer_2.requires_grad_(not(batch_norm_only))
        self.layer_3.requires_grad_(not(batch_norm_only))

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_loss = 0
        self.val_loss = 0
        self.first_test_done = False

        # Activation function used
        self.activation = torch.relu

        self.save_hyperparameters()

    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Dict[str, torch.tensor]:
        # first layer, batch norm and activation
        o = self.activation(
            self.batch_n1(
                self.layer_1(
                    x)))

        # second layer, batch norm and activation
        o = self.activation(
            self.batch_n2(
                self.layer_2(o)))

        # third layer, batch norm and activation
        o = self.activation(
            self.batch_n3(
                self.layer_3(o)))

        # final layer and softmax
        logits = self.layer_output(o)
        pred = self.softmax(logits)
        output = {"probabilities": pred, "logits": logits}
        if y is not None:  # if there is a ground truth, compute the loss
            loss = self.loss(logits, y)
            output["loss"] = loss
        return output

    def training_step(self, batch: Tuple[torch.tensor], batch_idx: int) -> torch.tensor:
        out = self.forward(*batch)
        self.train_loss += out["loss"].item()
        return out["loss"]

    def training_epoch_end(self, outputs: torch.tensor) -> None:
        epoch = self.trainer.current_epoch
        n_batches = len(outputs)
        wandb.log({"training_loss": self.train_loss/n_batches, "epoch": epoch})
        self.train_loss = 0

    def validation_step(self, batch: Tuple[torch.tensor], batch_idx: int) -> torch.tensor:
        out = self.forward(*batch)
        self.val_loss += out["loss"].item()
        return out["probabilities"], batch[1]

    def validation_epoch_end(self, outputs: torch.tensor) -> None:
        epoch = self.trainer.current_epoch
        n_batches = len(outputs)
        predictions = torch.cat([x[0] for x in outputs])
        labels = torch.cat([x[1] for x in outputs])
        accuracy = torchmetrics.functional.accuracy(predictions, labels).item()
        loss = self.val_loss/n_batches
        self.val_loss = 0
        self.log("val_loss", loss)
        wandb.log({"validation_loss": loss, "epoch": epoch}, commit=False)
        wandb.log({"validation_accuracy": accuracy, "epoch": epoch})

    def test_step(self, batch: Tuple[torch.tensor], batch_idx: int) -> torch.tensor:
        out = self.forward(*batch)
        return out["probabilities"], batch[1]

    def test_epoch_end(self, outputs: torch.tensor) -> None:
        predictions = torch.cat([x[0] for x in outputs])
        labels = torch.cat([x[1] for x in outputs])
        accuracy = torchmetrics.functional.accuracy(predictions, labels).item()
        if self.first_test_done:
            wandb.run.summary["final_test_accuracy"] = accuracy
            self.log("final_test_accuracy", accuracy)
        else:
            wandb.run.summary["initial_test_accuracy"] = accuracy
            self.log("initial_test_accuracy", accuracy)
            self.first_test_done = True

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def loss(self, prediction: torch.tensor, true: torch.tensor) -> torch.tensor:
        return self.loss_fn(prediction, true)

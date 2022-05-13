import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import wandb
from typing import *

class MLPModule(pl.LightningModule):
    def __init__(self, input_dim : int, hidden1_dim : int , hidden2_dim : int, hidden3_dim : int, n_classes : int) -> None:
        super().__init__()

        #Network structure
        self.batch_n0 = nn.BatchNorm1d(input_dim, affine = False) #the batch normalization at the beginning doesn't have learnable params (?) TODO
        self.layer_1 = nn.Linear(input_dim, hidden1_dim)
        self.batch_n1 = nn.BatchNorm1d(hidden1_dim)
        self.layer_2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.batch_n2 = nn.BatchNorm1d(hidden2_dim)
        self.layer_3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.batch_n3 = nn.BatchNorm1d(hidden3_dim)
        self.layer_output = nn.Linear(hidden3_dim, n_classes)
        self.softmax = nn.Softmax(dim  = 1)

        #Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        #Metrics
        self.val_metric = torchmetrics.Accuracy(num_classes = n_classes)
        self.test_metric = torchmetrics.Accuracy(num_classes = n_classes) #TODO probably just one is needed

        #Activation function used
        self.activation = torch.relu

        self.save_hyperparameters()
    
    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Dict[str,torch.tensor]:
        o = self.activation(
            self.batch_n1(
            self.layer_1(
            self.batch_n0(x))))

        o = self.activation(
            self.batch_n2(
            self.layer_2(o)))
        
        o = self.activation(
            self.batch_n3(
            self.layer_3(o)))
        
        logits = self.layer_output(o)
        pred = self.softmax(logits)
        output = { "probabilities" : pred, "logits" : logits }
        if y is not None:
            loss = self.loss(logits,y)
            output["loss"] = loss
        return output

    def training_step(self, batch : Tuple[torch.tensor], batch_idx : int) -> torch.tensor:
        out = self.forward(*batch)
        epoch = self.trainer.current_epoch
        self.log('step', epoch)
        self.log("Traning loss", out["loss"].item(), on_step = False ,on_epoch = True, prog_bar = True)
        wandb.log({"Training loss" : out["loss"].item(), "epoch" : epoch})
        return out["loss"]
    
    def validation_step(self, batch : Tuple[torch.tensor], batch_idx : int) -> torch.tensor:
        out = self.forward(*batch)
        epoch = self.trainer.current_epoch
        self.val_metric(out["probabilities"], batch[1]) #compute accuracy
        self.log('step', epoch)
        self.log("validation accuracy", self.val_metric, on_step = False ,on_epoch = True, prog_bar = True) #log result
        self.log("Validation loss",out["loss"], on_step = False, on_epoch = True, prog_bar = True)
        wandb.log({"Validation loss" : out["loss"].item(), "epoch" : epoch})
        wandb.log({"Validation accuracy" : self.val_metric, "epoch" : epoch})
        return out["loss"]

    def test_step(self, batch : Tuple[torch.tensor], batch_idx : int) -> torch.tensor:
        out = self.forward(*batch)
        self.test_metric(out["probabilities"], batch[1]) #compute accuracy
        self.log("Test accuracy", self.test_metric, on_epoch = True, prog_bar = True) #log result
        return out["loss"]
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())
        
    def loss(self, prediction : torch.tensor, true : torch.tensor) -> torch.tensor:
        return self.loss_fn(prediction,true)
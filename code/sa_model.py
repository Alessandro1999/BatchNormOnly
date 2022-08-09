'''
This file contains the code regarding the pytorch lightning model
of the test on SA with a LSTM
'''
from typing import *
import config
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import torchmetrics


class SentimentClassifier(pl.LightningModule):
    '''
    The LSTM model with batch normalization.
    - vocab_size, the size of the vocabulary of the "known" words;
    - embedding_dim, the dimension of a word embedding;
    - hidden_dim, the dimension of the hidden state of the LSTM;
    - bidirectional(Optional), True if the LSTM is bidirectional, False by default;
    - pretrained(Optional), if given, it is used to initialize the word embeddings;
    - padding_idx(Optional), the index assigned to the padding word, 0 by default. 
    '''

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 bidirectional: bool = False,
                 pretrained: torch.Tensor = None,
                 batch_norm_only: bool = False,
                 padding_idx: int = 0) -> None:
        super().__init__()

        multiplier = 2 if bidirectional else 1

        # network structure
        if pretrained is None:  # we initialize word embeddings randomly
            self.embedding = nn.Embedding(vocab_size,
                                          embedding_dim,
                                          padding_idx=padding_idx,
                                          )
            self.embedding.requires_grad_(not batch_norm_only)
        else:  # we initialize word embeddings with the pretrained ones
            self.embedding = nn.Embedding.from_pretrained(
                pretrained, freeze=batch_norm_only)

        self.bn1 = nn.BatchNorm1d(embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            bidirectional=bidirectional,
                            batch_first=True)

        self.bn2 = nn.BatchNorm1d(hidden_dim*multiplier)

        self.linear = nn.Linear(hidden_dim*multiplier,
                                n_classes)

        self.bn3 = nn.BatchNorm1d(n_classes)

        # activation & loss
        self.softmax = nn.Softmax(dim=-1)
        self.activation = torch.relu
        self.loss = nn.CrossEntropyLoss()

        # log attributes
        self.val_loss = 0
        self.train_loss = 0
        self.first_test_done = False

        # freeze the weights of the LSTM and the linear layer at random inizialization (if required)
        self.lstm.requires_grad_(not batch_norm_only)
        self.linear.requires_grad_(not batch_norm_only)

        self.save_hyperparameters()

    def forward(self, sentences: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # compute word embeddings (B x SL) -> (B x SL x EMB)
        embeddings = self.embedding(sentences)
        embeddings = torch.einsum(
            "bcl -> blc", self.bn1(torch.einsum("blc -> bcl", embeddings)))

        embeddings = self.activation(embeddings)

        # lstm (B x SL x EMB) -> (B x SL x HID)
        out, _ = self.lstm(embeddings)

        batch_size, _, _ = out.shape
        # take for each sample only the last output before the pad (B x SL x HID) -> (B x HID)
        out = out[torch.arange(batch_size), lengths-1, :]
        out = self.bn2(out)
        out = self.activation(out)

        output = dict()

        # linear layer to obtain the logits (B x HID) -> (B x N)
        output["logits"] = self.linear(out)
        output["logits"] = self.bn3(output["logits"])

        # obtain probabilities with the softmax
        output["probabilities"] = self.softmax(output["logits"])

        # if there is the ground truth, compute the loss
        if labels is not None:
            output["loss"] = self.loss(output["logits"], labels)

        return output

    def training_step(self, batch: Tuple[torch.tensor], batch_idx: int) -> torch.tensor:
        out = self.forward(**batch)
        self.train_loss += out["loss"].item()
        return out["loss"]

    def training_epoch_end(self, outputs: torch.tensor) -> None:
        epoch = self.trainer.current_epoch
        n_batches = len(outputs)
        wandb.log({"training_loss": self.train_loss/n_batches, "epoch": epoch})
        self.train_loss = 0

    def validation_step(self, batch: Tuple[torch.tensor], batch_idx: int) -> torch.tensor:
        out = self.forward(**batch)
        # self.val_metric(out["probabilities"], batch[1]) #compute accuracy
        self.val_loss += out["loss"].item()
        return out["probabilities"], batch["labels"]

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
        out = self.forward(**batch)
        return out["probabilities"], batch["labels"]

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

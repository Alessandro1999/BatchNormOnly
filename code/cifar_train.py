'''
This file contains the code used for the test on CIFAR-10 with a MLP
'''
from typing import *
import torch
import config
import wandb
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from torchvision import datasets
import utils
import cifar_datamodule
import cifar_model

# reproducibility
pl.seed_everything(config.seed, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# the test is on CIFAR-10; however, to switch to CIFAR-100, is sufficient to change this function
dataset_function: Callable = datasets.CIFAR10

# hyperparameters
batch_size = 32
width, height, channels = 32, 32, 3
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 256
n_classes = 10 if dataset_function == datasets.CIFAR10 else 100
batch_norm_only = True
epochs = 100

# the datamodule used for training and testing
datamodule = cifar_datamodule.CIFARDatamodule(
    dataset_function, batch_size, val_percentage=0.1)

# initialize the model
model = cifar_model.MLPModule(width*height*channels,
                              hidden_dim1,
                              hidden_dim2,
                              hidden_dim3,
                              n_classes,
                              batch_norm_only)

# Weights and Biases initialization
wandb.init(project="Batch-norm-only", entity="ale99")
wandb.run.name = f"CIFAR{n_classes} First trial"
wandb.define_metric("epoch")
wandb.define_metric("validation_loss", step_metric="epoch", summary="min")
wandb.define_metric("validation_accuracy", step_metric="epoch", summary="max")
wandb.define_metric("training_loss", step_metric="epoch", summary="min")

# Checkpoint to save the model with the lowest validation loss
checkpoint = callbacks.ModelCheckpoint("checkpoints/",
                                       monitor="val_loss",
                                       mode="min")

# Pytorch lightning trainer
trainer = pl.Trainer(max_epochs=epochs,
                     gpus=1,
                     callbacks=[checkpoint])

# first test before the training (I expect an accuracy around 10%)
trainer.test(model=model, datamodule=datamodule)

# training
trainer.fit(model=model, datamodule=datamodule)

# load the model with the lowest validation loss
model = cifar_model.MLPModule.load_from_checkpoint(
    checkpoint.best_model_path)
model.first_test_done = True

# final evaluation on the test set
trainer.test(model=model, datamodule=datamodule)

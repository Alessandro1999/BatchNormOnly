'''
This file contains the code used for the test on SA with a LSTM
'''
import torch
import config
from pathlib import Path
import wandb
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import gensim.downloader
from gensim.models import KeyedVectors
import sa_datamodule
import sa_model
import utils

# reproducibility
pl.seed_everything(config.seed, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# paths to the datasets and the pretrained embeddings
train_path: Path = config.ROOT_PATH.joinpath(
    "data/SA/training_set.csv")
test_path: Path = config.ROOT_PATH.joinpath(
    "data/SA/test_set.csv")

embedding_path: Path = config.ROOT_PATH.joinpath(
    "data/SA/Embeddings/glove.wordvectors")

# hyperparameters
batch_size = 32
min_count = 2
embedding_dim = 200
hidden_dim = 100
bidirectional = False
batch_norm_only = True
pretrained = True
embeddings: torch.Tensor = None
epochs = 100

# the datamodule used for training and testing
datamodule = sa_datamodule.SentimentDataModule(train_path,
                                               test_path,
                                               train_split=0.1,
                                               train_batch_size=batch_size,
                                               min_count=min_count)


# if specified in the hyperparameters, pretrained embeddings are loaded
if pretrained:
    # Pretrained GloVe embeddings
    try:  # try to load them
        glove: KeyedVectors = KeyedVectors.load(str(embedding_path))
    except:  # otherwise, download them from gensim
        glove: KeyedVectors = gensim.downloader.load("glove-twitter-200")
        glove.load(str(embedding_path))
    finally:
        embeddings = utils.build_pretrained_vectors(glove)

# initialize the model
model = sa_model.SentimentClassifier(len(config.word2id),
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim,
                                     n_classes=2,
                                     bidirectional=bidirectional,
                                     batch_norm_only=batch_norm_only,
                                     pretrained=embeddings,
                                     padding_idx=config.word2id[config.PAD_WORD])

# Weights and Biases initialization
wandb.init(project="Batch-norm-only", entity="ale99")
wandb.run.name = "[BNO]EmbRELU-NoDrop"
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

# first test before the training (I expect an accuracy around 50%)
trainer.test(model=model, datamodule=datamodule)

# training
trainer.fit(model=model, datamodule=datamodule)

# load the model with the lowest validation loss
model = sa_model.SentimentClassifier.load_from_checkpoint(
    checkpoint.best_model_path)
model.first_test_done = True

# final evaluation on the test set
trainer.test(model=model, datamodule=datamodule)

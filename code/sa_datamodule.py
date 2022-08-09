'''
This file contains the code regarding the pytorch lightning datamodule
of the test on SA with a LSTM
'''
import config
import utils
from typing import *
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sa_dataset import SentimentDataset


class SentimentDataModule(pl.LightningDataModule):
    '''
    The datamoodule used for the SA test with LSTM
    - train_path, the training set path;
    - test_path, the test set path;
    - train_split, the percentage of training set used for the validation set;
    - train_batch_size, the batch size used for the training set;
    - valid_batch_size(Optional), the batch size for test and validation set, if not given, train_batch_size is used;
    - min_count, the minimum number of occurrencies in the training set for a word to have its own word embedding. 
    '''

    def __init__(self,
                 train_path: Path,
                 test_path: Path,
                 train_split: float = 0.2,
                 train_batch_size: int = 32,
                 valid_batch_size: int = None,
                 min_count: int = 2) -> None:
        super().__init__()

        # train and validation dataframes
        train_df: pd.DataFrame = utils.load_SA_data(train_path)
        train_df, val_df = train_test_split(
            train_df, test_size=train_split, random_state=config.seed)

        # test set dataframe
        test_df = utils.load_SA_data(test_path)

        # datasets obtained from the dataframe; also the vocabulary is computed
        self.train_data = SentimentDataset(train_df,
                                           is_train=True,
                                           min_count=min_count)

        self.val_data = SentimentDataset(val_df,
                                         is_train=False,
                                         min_count=min_count)

        self.test_data = SentimentDataset(test_df,
                                          is_train=False)

        self.train_batch_size = train_batch_size
        self.valid_batch_size = train_batch_size if valid_batch_size is None else valid_batch_size

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.test_data, batch_size=self.valid_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.val_data, batch_size=self.valid_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)


def collate_fn(batch: List[Tuple[List[str], int]]) -> Dict[str, torch.Tensor]:
    '''
    Function used by Dataloaders to convert (and pad) sentences into indexes tensors
    '''
    labels: torch.Tensor = torch.zeros(len(batch), dtype=torch.long)

    # the length of each sentence of the batch (not padded)
    lengths: torch.Tensor = torch.zeros(len(batch), dtype=torch.long)

    sentences: List[torch.Tensor] = []
    for i, (sentence, label) in enumerate(batch):  # for every sample of the batch
        labels[i] = label  # assign the label
        lengths[i] = len(sentence)  # obtain the length
        sentence_tensor = torch.zeros(len(sentence), dtype=torch.int32)
        # replace each word with its index in the vocabulary
        for w, word in enumerate(sentence):
            word = word.replace(",", " ").replace(".", " ").replace("!", " ").replace(
                "?", " ").replace("'", " ").replace("(", " ").replace(")", " ").replace("=", " ").lower()
            sentence_tensor[w] = config.word2id.get(
                word, config.word2id[config.UNK_WORD])  # convert the sentence into indexes

        sentences.append(sentence_tensor)
    out: Dict[str, torch.Tensor] = {"labels": labels, "lengths": lengths}

    out["sentences"] = torch.nn.utils.rnn.pad_sequence(  # pad the sentences to make all of them of the same length
        sentences,
        batch_first=True,
        padding_value=config.word2id[config.PAD_WORD])
    return out

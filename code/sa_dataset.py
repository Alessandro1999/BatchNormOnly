'''
This file contains the code to define the dataset used for SA
'''
from typing import *
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import config
from torch.utils import data


class SentimentDataset(data.Dataset):
    '''
    Dataset for the sentiment analysis task.
    - df, the dataframe of the dataset;
    - is_train, True if it is the training set, False otherwise;
    - min_count, the minimum number of occurrencies of a word in the training set to have its own word embedding.
    '''

    def __init__(self,
                 df: pd.DataFrame,
                 is_train: bool,
                 min_count: int = 2) -> None:
        super().__init__()

        # obtain sentences and their polarity
        self.sentences: List[List[str]] = df.Text.replace(
            "[.,!?'()=]", " ", regex=True).str.lower().str.split().values

        self.labels: List[int] = df.Polarity.values

        # for the training set, we also compute a dictionary of words
        if is_train:
            config.word2id: Dict[str, int] = self.get_vocab(
                min_count=min_count)
            config.id2word: Dict[int, str] = {
                idx: word for word, idx in config.word2id.items()}

    def get_vocab(self, min_count: int = 2) -> Dict[str, int]:
        '''
        This function assigns a unique index to every word that appears at list
        min_count times in the training set.
        '''
        counter: Dict[str, int] = dict()
        # we count the occurrencies of each word in the dataset
        for sentence in tqdm(self.sentences, total=len(self.sentences), desc="Computing vocabulary..."):
            for word in sentence:
                counter[word] = counter.get(word, 0) + 1

        # we keep only the words appearing at least min_count times
        words: Set[str] = {
            word for word in counter if counter[word] >= min_count}

        # we map each word to a unique index
        word2id: Dict[str, int] = {
            word: idx + 2 for idx, word in enumerate(sorted(words))}

        # index 0 and 1 are reserved for the padding word and for the out of vocabulary word
        word2id[config.PAD_WORD] = 0
        word2id[config.UNK_WORD] = 1

        return word2id

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[List[str], int]:
        return self.sentences[index], self.labels[index]

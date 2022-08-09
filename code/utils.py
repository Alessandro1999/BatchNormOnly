'''
This file contains some utility function to load dataframes or plot the batch norm distributions
'''
from pathlib import Path
from typing import *
from tqdm import tqdm
import pandas as pd
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import torch
import config


def load_SA_data(data_path: Path) -> pd.DataFrame:
    '''
    This function loads the pandas dataframe from the data_path given.
    '''
    df: pd.DataFrame = pd.read_csv(data_path,
                                   encoding="ISO-8859-1",
                                   names=["Polarity", "ID", "Date", "Query", "User", "Text"])

    # polarity becomes : 0 -> negative, 1 -> positive
    df.Polarity = df.apply(lambda row: row.Polarity // 4, axis=1)

    return df


def build_pretrained_vectors(word_vectors: KeyedVectors) -> torch.Tensor:
    '''
    This function loads the pretrained embedding into a pytorch tensor of shape (vocab_size, embedding_dim) 
    '''
    out = torch.rand(len(config.word2id), word_vectors.vector_size)
    ids = []
    # for every word in the vocabulary
    for word, idx in tqdm(config.word2id.items(), total=len(config.word2id), desc="Building pretrained embeddings"):
        # if we have the embedding for the word, we add it; otherwise, we keep it at its random initialitazion
        if word != config.PAD_WORD and word != config.UNK_WORD and word in word_vectors:
            out[idx] = torch.tensor(word_vectors[word])
            ids.append(idx)

    # pad embedding is initialized as all 0s
    out[config.word2id[config.PAD_WORD]] = torch.zeros(
        word_vectors.vector_size)

    # unk embedding is initialized as the mean of all the others
    out[config.word2id[config.UNK_WORD]] = out[ids, :].mean(dim=0)

    return out


def gamma_values(model: torch.nn.Module, model_name: str = "Model") -> Tuple[Dict[float, int], float, float]:
    '''
    Given a model that uses 1d batch normalization, this function plots the distribution of the
    gamma values, along with the mean and standard deviation
    '''
    gamma_dict: Dict[float, int] = dict()
    mean: float = 0
    std: float = 0
    n: int = 0
    # for every layer of the model
    for layer in model.modules():
        # if the layer is a batch norm layer (1d)
        if isinstance(layer, torch.nn.modules.BatchNorm1d):
            gammas: torch.Tensor = layer.weight.flatten()  # obtain the gamma values
            for value in gammas:  # store them and compute the mean
                n += 1
                v = round(value.item(), 1)
                mean += v
                gamma_dict[v] = gamma_dict.get(v, 0) + 1

    mean = round(mean/n, 2)
    # compute the standard deviation
    for value, times in gamma_dict.items():
        for _ in range(times):
            std += (value - mean)**2
    std = round((std/n)**0.5, 2)

    # plot the distribution
    plt.title(
        f"{model_name}: Gamma distribution (Mean = {mean}, Std = {std}) ")
    plt.bar(gamma_dict.keys(), gamma_dict.values())
    plt.show()
    return gamma_dict, mean, std


def beta_values(model: torch.nn.Module, model_name="Model") -> Tuple[Dict[float, int], float, float]:
    '''
    Given a model that uses 1d batch normalization, this function plots the distribution of the
    beta values, along with the mean and standard deviation
    '''
    beta_dict: Dict[float, int] = dict()
    mean: float = 0
    std: float = 0
    n: int = 0
    # for every layer of the model
    for layer in model.modules():
        # if the layer is a batch norm layer (1d)
        if isinstance(layer, torch.nn.modules.BatchNorm1d):
            betas: torch.Tensor = layer.bias.flatten()  # obtain the beta values
            for value in betas:  # store them and compute the mean
                n += 1
                v = round(value.item(), 1)
                mean += v
                beta_dict[v] = beta_dict.get(v, 0) + 1

    mean = round(mean/n, 2)
    # compute the standard deviation
    for value, times in beta_dict.items():
        for _ in range(times):
            std += (value - mean)**2
    std = round((std/n)**0.5, 2)

    # plot the distribution
    plt.title(f"{model_name}: Beta distribution (Mean = {mean}, Std = {std}) ")
    plt.bar(beta_dict.keys(), beta_dict.values())
    plt.show()
    return beta_dict, mean, std

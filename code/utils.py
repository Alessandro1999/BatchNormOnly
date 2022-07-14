from pathlib import Path
from tqdm import tqdm
import pandas as pd
from gensim.models import KeyedVectors
import torch
import config


def load_SA_data(data_path: Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(data_path,
                                   encoding="ISO-8859-1",
                                   names=["Polarity", "ID", "Date", "Query", "User", "Text"])

    # polarity becomes : 0 -> negative, 1 -> positive
    df.Polarity = df.apply(lambda row: row.Polarity // 4, axis=1)

    #df.drop(columns=["Date", "Query","ID","User"])
    return df


def build_pretrained_vectors(word_vectors: KeyedVectors) -> torch.Tensor:
    out = torch.rand(len(config.word2id), word_vectors.vector_size)
    ids = []
    for word, idx in tqdm(config.word2id.items(), total=len(config.word2id), desc="Building pretrained embeddings"):
        if word != config.PAD_WORD and word != config.UNK_WORD and word in word_vectors:
            out[idx] = torch.tensor(word_vectors[word])
            ids.append(idx)

    # pad embedding is initialized as all 0s
    out[config.word2id[config.PAD_WORD]] = torch.zeros(
        word_vectors.vector_size)

    # unk embedding is initialized as the mean of all the others
    out[config.word2id[config.UNK_WORD]] = out[ids, :].mean(dim=0)

    return out

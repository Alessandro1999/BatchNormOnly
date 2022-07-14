from pathlib import Path
import os
import sys
import torch
from typing import Dict

ROOT_PATH: Path = Path(__file__).parent.parent
sys.path.append("code/")

UNK_WORD: str = "<UNK>"
PAD_WORD: str = "<PAD>"
word2id: Dict[str, int] = dict()
id2word: Dict[int, str] = dict()
pretrained_embeddings: torch.Tensor = None

seed = 17
os.chdir(ROOT_PATH)

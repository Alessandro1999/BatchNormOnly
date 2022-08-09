'''
This file is used to make the project runnable
indipendently from where it is run; also, it keeps
some shared variable between files
'''
from pathlib import Path
import os
import sys
import torch
from typing import Dict

# the root path of the project
ROOT_PATH: Path = Path(__file__).parent.parent
sys.path.append("code/")

# NLP variables
UNK_WORD: str = "<UNK>"
PAD_WORD: str = "<PAD>"
word2id: Dict[str, int] = dict()
id2word: Dict[int, str] = dict()
pretrained_embeddings: torch.Tensor = None

# seed used
seed = 17

# the working directory is set to the root path
os.chdir(ROOT_PATH)

from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms,datasets
from typing import *
from tqdm import tqdm

#linearize = lambda t : t.reshape(-1) TODO (?)
image_transforms =  transforms.Compose(
                    [ 
                        transforms.ToTensor(), #convert from PIL image to tensor
                        transforms.Lambda(torch.flatten) #flatten the tensor
                    ]
                    )


def count_parameters(model : torch.nn.Module) -> int:
    '''
    Given a pytorch model, it returns the number of parameters it has
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_cifar10() -> Tuple[DataLoader,DataLoader,DataLoader]:
    '''
    This function returns a dataloader for the training, validation and test set of the dataset CIFAR-10
    '''
    cifar10_train_set = datasets.CIFAR10("data/", download = True, transform = image_transforms)
    cifar10_test_set, cifar10_validation_set = random_split(datasets.CIFAR10("data/", train = False ,download = True, transform = image_transforms), [5000,5000])
    return cifar10_train_set, cifar10_validation_set, cifar10_test_set

def get_cifar100() -> Tuple[DataLoader,DataLoader,DataLoader]:
    '''
    This function returns a dataloader for the training, validation and test set of the dataset CIFAR-100
    '''
    cifar100_train_set = datasets.CIFAR100("data/", download = True, transform = image_transforms)
    test = datasets.CIFAR100("data/", train = False ,download = True, transform = image_transforms)
    cifar100_test_set, cifar100_validation_set = random_split(test, [5000,5000])
    return cifar100_train_set, cifar100_validation_set, cifar100_test_set

#TODO cite http://ai.stanford.edu/~amaas/data/sentiment/
def get_imdb() -> Tuple[DataLoader, DataLoader, DataLoader]:
    datafolder = os.sep + "data"+ os.sep + "aclImdb" + os.sep
    path_train = datafolder + "train" + os.sep
    path_test = datafolder + "test" + os.sep 
    train_sentences, train_labels = read_imdb(path_train)
    test_sentences, test_labels = read_imdb(path_test)

def read_imdb(path : str):
    sentences, labels = read_dir(path + "pos",1)
    sentences_neg, labels_neg = read_dir(path + "neg",0)
    sentences = sentences + sentences_neg
    labels = torch.tensor(labels + labels_neg)
    return sentences,labels

def read_dir(dir : str, sentiment : int):
    sentences = []
    labels = []
    for example in tqdm(os.listdir(dir), desc = f"Reading sentences from:  {dir} "):
        with open(Path(dir+os.sep+example),"r",encoding = "utf-8") as ex:
            for line in ex:
                sentences.append(line)
                labels.append(sentiment)
    return sentences, labels



cifar10_params = [3072,1024,128,32,10] #layers size of cifar10 model
cifar100_params = [3072,1024,512,256,100]  #layers size of cifar100 model
imdb_params = []

dataset_options : Dict[str, Tuple[Callable,List]] = {
    "cifar-10" : (get_cifar10, cifar10_params),
    "cifar-100" : (get_cifar100, cifar100_params),
    "imdb" : (get_imdb, imdb_params)
}
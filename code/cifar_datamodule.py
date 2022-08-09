'''
This file contains the code regarding the pytorch lightning datamodule
of the test on CIFAR-10 with a MLP
'''
from typing import *
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

'''
Image transformations:
First, the PIL image is converted into a pytorch tensor and then
it is linearized into a one-dimensional tensor.
'''
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),  # convert from PIL image to tensor
        transforms.Lambda(torch.flatten)  # flatten the tensor
    ]
)


class CIFARDatamodule(pl.LightningDataModule):
    '''
    The datamoodule used for the CIFAR-10 test on the MLP
    - dataset_function, the function used to obtain the CIFAR dataset;
    - train_bs, the batch size for the training set;
    - valid_bs(Optional), the batch size for test and validation set, if not given, train_bs is used;
    - val_percentage, the percentage of samples from the training set to use as validation set.
    '''

    def __init__(self, dataset_function: Callable, train_bs: int, valid_bs: int = None, val_percentage: float = 0.1):
        super().__init__()

        self.dataset_function = dataset_function
        self.val_percentage = val_percentage
        self.train_data: Dataset = None
        self.valid_data: Dataset = None
        self.test_data: Dataset = None
        self.train_batch_size = train_bs
        self.valid_batch_size = train_bs if valid_bs is None else valid_bs

    def setup(self, stage: Optional[str] = None) -> None:
        # the train dataset
        train = self.dataset_function(
            "data/", download=True, transform=image_transforms)

        val_num: int = int(len(train) * self.val_percentage)

        # train/validation split
        self.train_data, self.valid_data = random_split(
            train, [len(train)-val_num, val_num])

        # the test set
        self.test_data = self.dataset_function(
            "data/", train=False, download=True, transform=image_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.valid_batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, batch_size=self.valid_batch_size, shuffle=False, num_workers=4)

from typing import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class Datamodule(pl.LightningDataModule):
    '''
    A pytorch datamodule for a generic dataset
    '''
    def __init__(self, dataset_name : str, train_batch_size : int, test_batch_size : int, dataset_options : Dict[str,Tuple[Callable, List]]):
        '''
        Parameters:
            - dataset_name, the name of the dataset you want to load
            - train_batch_size, the lenght of the batch size of the training dataloader
            - test_batch_size, the lenght of the batch size of the test and validation dataloaders
            - dataset_option, a dictionary that contains the function to load the dataset (given the dataset name)
        '''
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.dataset_name = dataset_name
        self.dataset_options = dataset_options
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        Loads the data using the function defined in the dataset_option dictionary 
        '''
        try:
            self.train_data, self.val_data, self.test_data = self.dataset_options[self.dataset_name][0]()
        except:
            print("Error, invalid dataset name: {}".format(self.dataset_name))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size = self.train_batch_size, shuffle = True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size = self.test_batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size = self.test_batch_size)
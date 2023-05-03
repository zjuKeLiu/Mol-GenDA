from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class TrajDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_csv(path, index_col=0).values
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index]

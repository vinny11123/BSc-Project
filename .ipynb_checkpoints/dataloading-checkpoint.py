import torch
from torch.utils.data import Dataset

class ChessDataSet(Dataset):

    def __init__(self, data, transform=None):
        """
        Argments:
            data (string): pass in data rows including labels.
            transform (callable, optional): Optional transform that can be applied to each row of chess data.
        """
        self.chess_data = data
        self.transform = transform

    def __len__(self):
        return len(self.chess_data)

    def __getitem__(self, idx): 
        features = self.chess_data[idx][:-1]
        label = self.chess_data[idx][-1]
        sample = {"features":torch.Tensor(features), "label":label}
        return sample
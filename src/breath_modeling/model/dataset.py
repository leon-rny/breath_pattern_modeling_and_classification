import torch
from torch.utils.data import Dataset

class BreathDataset(Dataset):
    """A PyTorch Dataset for breath data."""
    def __init__(self, X, y):
        """
        Initializes the BreathDataset with features and labels.
        :param X: Features, can be a list or a numpy array.
        :param y: Labels, can be a list or a numpy array.
        Converts both X and y to PyTorch tensors if they are not already.
        """
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """        Retrieves a sample from the dataset.
        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the features and label of the sample at index idx.
        """
        return self.X[idx], self.y[idx]


class BreathSequenceDataset(Dataset):
    """A PyTorch Dataset for breath sequences."""
    def __init__(self, X, y):
        self.X = X  
        self.y = y 

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset.
        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the features and label of the sample at index idx.
        """
        return self.X[idx], self.y[idx]
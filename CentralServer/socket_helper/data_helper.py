import torch as th
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import syft as sy
import pandas as pd

class FashionDataset(Dataset):
    def __init__(self, train_csv_path, transform = None):
        self.data = pd.read_csv(train_csv_path)
        self.transform = transform

        self.images = list()
        self.labels = list()

        for i in range(len(self.data)):
            self.images.append(self.transform(self.data.iloc[i, 1:].values.astype(np.uint8).reshape(1, 784)))
            self.labels.append(self.data.iloc[i, 0])

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.data)

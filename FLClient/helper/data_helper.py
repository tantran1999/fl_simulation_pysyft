import numpy as np
import pandas as pd
import torch as th
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class FashionDataset(Dataset):
    def __init__(self, train_csv_path, transform = None, isPoisoned = False, source_class=None, target_class=None):
        self.data = pd.read_csv(train_csv_path)
        self.transform = transform

        self.images = list()
        self.labels = list()
        
        for i in range(len(self.data)):
            self.images.append(self.transform(self.data.iloc[i, 1:].values.astype(np.uint8).reshape(1, 784)))
            self.labels.append(self.data.iloc[i, 0])
        
        if isPoisoned:
            self.replace_X_with_Y(source_class, target_class)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
    def replace_X_with_Y(self, source, target):
        for idx, label in enumerate(self.labels):
            if label == source:
                self.labels[idx] = target


def TrainLoader(train_csv_path, batch_size, poisoned, source_class, target_class) -> DataLoader:
    train_set = FashionDataset(train_csv_path, transforms.ToTensor(), poisoned, source_class, target_class)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

# def TrainLoaderTest(batch_size) -> DataLoader:
#     train_data = th.utils.data.DataLoader(
#         datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
#         batch_size=batch_size,
#         drop_last=True,
#         shuffle=True
#     )
#     return train_data

# train_loader = TrainLoader("/home/tantran/Documents/Model-centric-FL/FLClient/dataset/client_1_train_data.csv", 64)

# ite = iter(train_loader)



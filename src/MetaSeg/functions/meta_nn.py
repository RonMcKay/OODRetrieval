import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MetricDataset(Dataset):
    def __init__(self, data):
        super(MetricDataset, self).__init__()
        self.data = data[0].squeeze()
        self.targets = data[1].squeeze()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float().flatten(), torch.tensor(self.targets[index]).float().flatten()

    def __len__(self):
        return self.data.shape[0]


class MetaNN(nn.Module):
    def __init__(self, input_size):
        super(MetaNN, self).__init__()
        self.act = nn.ReLU()
        self.layers = nn.Sequential(nn.Linear(input_size, 50),
                                    self.act,
                                    nn.Linear(50, 40),
                                    self.act,
                                    nn.Linear(40, 30),
                                    self.act,
                                    nn.Linear(30, 20),
                                    self.act,
                                    nn.Linear(20, 10),
                                    self.act,
                                    nn.Linear(10, 1))

    def forward(self, x):
        return self.layers(x).view(x.shape[0], -1)

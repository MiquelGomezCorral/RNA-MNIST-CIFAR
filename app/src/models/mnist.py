import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MnistNet(nn.Module):
    def __init__(self, num_classes):
        super(MnistNet, self).__init__()
        self.linear1 = nn.Linear(784, 1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.relu1(self.linear1(x))
        out = self.relu2(self.linear2(out))
        out = self.relu3(self.linear3(out))
        out = self.classifier(out)
        return out
import torch.nn as nn

class MnistNet(nn.Module):
    def __init__(self, num_classes):
        super(MnistNet, self).__init__()
        self.linear1 = nn.Linear(784, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.classifier(out)
        return out
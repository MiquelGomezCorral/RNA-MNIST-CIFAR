import torch.nn as nn
import torch

class CifarNet(nn.Module):
    def __init__(self, num_classes=10, channel_sizes=[3, 32, 64, 128, 256, 512]):
        super(CifarNet, self).__init__()
        self.conv_layers = nn.Sequential(*[
            ConvBlock(channel_sizes[i], channel_sizes[i+1]) for i, _ in enumerate(channel_sizes[:-1])
        ])

        self.ff1 = nn.Linear(channel_sizes[-1], channel_sizes[-1])
        self.ff2 = nn.Linear(channel_sizes[-1], 256)
        self.ff3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.ff1(x))
        x = self.relu(self.ff2(x))
        x = self.ff3(x)

        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
import torch.nn as nn
import torch

class CifarNet(nn.Module):
    def __init__(self, num_classes=10, channel_sizes=[3, 64, 128, 512], dropout_rate=0.5):
        super(CifarNet, self).__init__()
        self.conv_layers = nn.Sequential(*[
            ResNetBlock(channel_sizes[i], channel_sizes[i+1]) for i, _ in enumerate(channel_sizes[:-1])
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ff1 = nn.Linear(channel_sizes[-1], channel_sizes[-1])
        self.ff2 = nn.Linear(channel_sizes[-1], 128)
        self.ff3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.ff1(x)))
        x = self.dropout(self.relu(self.ff2(x)))
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
    
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        
        self.relu = nn.ReLU()

    def forward(self, x):
        previous_x = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        previous_x = self.conv_shortcut(previous_x)

        out += previous_x
        out = self.relu(out)

        return out
import torch.nn as nn

class MnistNet(nn.Module):
    def __init__(self, num_classes):
        super(MnistNet, self).__init__()
        self.sizes=[
            [784, 1536],
            [1536, 1536],
            [1536, 1536],
            [1536, 512],
            [512, 256],
            [256, num_classes]
        ]
        self.layers = nn.ModuleList()
        self.skip_size = self.sizes[0][0]  # Tamaño de la entrada inicial 

        for i in range(len(self.sizes)-1):
            dims = self.sizes[i]
            if i % 3 == 1: # Cada 2 capas
                self.layers.append(
                    SkipConnection(dims[0], self.skip_size, dims[1])
                )
            else:
                self.layers.append(CustomLayer(dims[0], dims[1]))


        dims = self.sizes[-1]
        self.classifier = nn.Linear(dims[0], dims[1])

    def forward(self, x):
        skip = x.clone()
        for layer in self.layers:
            x = layer(x, skip)

        x = self.classifier(x)

        return x
    
class CustomLayer(nn.Module):
    def __init__(self, input, output):
        super(CustomLayer, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.BatchNorm1d(input))
        self.layers.append(nn.Linear(input, output))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.45))

    def forward(self, x, skip=None):
        for layer in self.layers:
            x = layer(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, input, skip_input, output):
        super(SkipConnection, self).__init__()
        self.batchnorm = nn.BatchNorm1d(input)
        self.linear = nn.Linear(input, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.45)

        self.batchnorm_skip = nn.BatchNorm1d(skip_input)
        self.linear_skip = nn.Linear(skip_input, output)

    def forward(self, x, skip):
        x = self.batchnorm(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)

        skip = self.batchnorm_skip(skip)
        skip = self.linear_skip(skip)
        x += skip
        x = self.relu(x)

        return x
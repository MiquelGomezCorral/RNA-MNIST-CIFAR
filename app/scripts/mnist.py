import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Configuration
from src.data import load_mnist
from src.models import MnistNet 
from src.utils import count_parameters, get_device
from src.models import train_model

def train_mnist(CONFIG: Configuration):
    # ===================== DATA =====================
    train_loader, test_loader = load_mnist(CONFIG)

    # ===================== MODEL ====================
    model = MnistNet(CONFIG.num_classes)
    count_parameters(model)

    CONFIG.device = get_device()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay, momentum=CONFIG.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs)


    # ==================== TRAINING ====================
    trained_model = train_model(
        CONFIG=CONFIG, 
        model=model, 
        train_dataloader=train_loader, 
        test_dataloader=test_loader, 
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler,
    )

  
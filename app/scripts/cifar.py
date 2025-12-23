import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Configuration
from src.data import load_cifar
from src.models import CifarNet 
from src.utils import count_parameters, get_device
from src.models import train_model

def train_cifar(CONFIG: Configuration):
    # ===================== DATA =====================
    train_loader, test_loader = load_cifar(CONFIG)

    # ===================== MODEL ====================
    model = CifarNet(CONFIG.num_classes, dropout_rate=CONFIG.dropout_rate)
    count_parameters(model)

    CONFIG.device = get_device()
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)
    # optimizer = optim.SGD(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay, momentum=CONFIG.momentum, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs)
    # Cosine Annealing with warm restarts
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.epochs//3, T_mult=1, eta_min=1e-6)

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

  
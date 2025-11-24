import torch
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.models import MnistNet

def train_model(
    CONFIG: Configuration,
    model: MnistNet, 
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
):
    """Train a model with given data.

    Args:
        CONFIG (Configuration): Configuration.
        model (MnistNet): Model to be trained.
        train_dataloader (DataLoader): training data loader.
        test_dataloader (DataLoader): testing data loader.
        criterion (nn.Module): loss function.
        optimizer (optim.Optimizer): optimization algorithm.

    Returns:
        models: MnistNet
    """
    print_separator("TRAINING", sep_type="LONG")
    model.to(CONFIG.device)
    best_accuracy, best_epoch = -1, 0

    # =================================================================================
    #                                       Training Loop 
    # =================================================================================
    for epoch in range(CONFIG.epochs):
        # =============== TRAIN NETWORK  ===============
        train_loss, train_correct = 0, 0
        model.train()
        with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for batch in tepoch:
                # Returned values of Dataset Class
                images = batch["img"].to(CONFIG.device)
                labels = batch["label"].to(CONFIG.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Calculate gradients
                loss.backward()

                # Update gradients
                optimizer.step()

                # one hot -> labels
                labels = torch.argmax(labels, dim=1)
                pred = torch.argmax(outputs, dim=1)
                train_correct += pred.eq(labels).sum().item()

                # print statistics
                train_loss += loss.item()

        train_loss /= len(train_dataloader.dataset)

        # =============== TEST NETWORK ===============
        test_loss, test_correct = 0, 0
        model.eval()
        with torch.no_grad():
            with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
                for batch in tepoch:
                    images = batch["img"].to(CONFIG.device)
                    labels = batch["label"].to(CONFIG.device)

                    # Forward
                    outputs = model(images)
                    test_loss += criterion(outputs, labels)
                    # one hot -> labels
                    labels = torch.argmax(labels, dim=1)
                    pred = torch.argmax(outputs, dim=1)

                    test_correct += pred.eq(labels).sum().item()

        test_loss /= len(test_dataloader.dataset)
        test_accuracy = 100. * test_correct / len(test_dataloader.dataset)

        print("[Epoch {}] Train Loss: {:.6f} - Test Loss: {:.6f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%".format(
            epoch + 1, train_loss, test_loss, 100. * train_correct / len(train_dataloader.dataset), test_accuracy
        ))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch

            # Save best weights
            torch.save(model.state_dict(), CONFIG.best_model_path)
    
    # NOTE: end for
    print("\nBEST TEST ACCURACY: ", best_accuracy, " in epoch ", best_epoch)

    # ================================================================================= 
    #                                    Final Evaluation
    # ================================================================================= 
    # Load best weights
    model.load_state_dict(torch.load(CONFIG.best_model_path))

    test_loss, test_correct = 0, 0
    model.eval()
    with torch.no_grad():
        with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
            for batch in tepoch:
                images = batch["img"].to(CONFIG.device)
                labels = batch["label"].to(CONFIG.device)

                # Forward
                outputs = model(images)
                test_loss += criterion(outputs, labels)

                # one hot -> labels
                labels = torch.argmax(labels, dim=1)
                pred = torch.argmax(outputs, dim=1)

                test_correct += pred.eq(labels).sum().item()

        test_loss /= len(test_dataloader.dataset)
        test_accuracy = 100. * test_correct / len(test_dataloader.dataset)
    print("Final best acc: ", test_accuracy)

    return model
"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass
import multiprocessing

@dataclass 
class Configuration:
    """Configuration class for the project."""
    # =============== Paths =============== 
    DATA_FOLDER: str = "../data/"
    MODEL_FOLDER: str = "../models/"
    LOGS_FOLDER: str = "../logs/"
    PID = None

    best_model_path: str = os.path.join(MODEL_FOLDER, "best_model.pth")

    # =============== Variabless =============== 
    description: str = "MNIST classification with MLP using Pytorch"
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda"
    num_workers: int = multiprocessing.cpu_count()-4
    num_classes: int = 10
    batch_size: int = 1024
    epochs: int = 200
    lr: float = 0.002
    weight_decay: float = 1e-6
    momentum: float = 0.9
    dropout_rate: float = 0.5

    # =============== Data aug parameters =============== 
    image_size: int = 28
    aug_prob: float = 0.5
    angle_range: int = 15  # degres
    translate_range: float = 0.1  # fraction of img size
    scale_range: tuple = (0.9, 1.1)
    shear_range: int = 10  # degres

    def __post_init__(self):
        self.PID = os.getpid()
        self.best_model_path = os.path.join(self.MODEL_FOLDER, f"best_model-{self.PID}.pth")

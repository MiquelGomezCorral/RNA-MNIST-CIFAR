import os 
import torch
import dataclasses
from torchinfo import summary

from src.models import MnistNet
from src.config import Configuration

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total}")
    return total


def save_config_score(CONFIG: Configuration, score: float, model: MnistNet) -> None:
    filepath = os.path.join(CONFIG.LOGS_FOLDER, f"config_score_{score:.4f}.txt")
    with open(filepath, "w") as f:
        print(CONFIG.description, file=f)
        for field in dataclasses.fields(CONFIG):
            value = getattr(CONFIG, field.name)
            f.write(f"{field.name}: {value}\n")
        
        model_summary = summary(model, input_size=(32, 784))
        print(model_summary, file=f)  
        
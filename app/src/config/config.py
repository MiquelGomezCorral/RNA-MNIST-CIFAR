"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
import dataclasses
from dataclasses import dataclass
from argparse import Namespace
import multiprocessing

@dataclass 
class Configuration:
    """Configuration class for the project."""
    DATA_FOLDER: str = "../data/"
    MODEL_FOLDER: str = "../models/"
    best_model_path: str = os.path.join(MODEL_FOLDER, "best_model.pth")

    seed: int = 42
    batch_size: int = 1024
    num_workers: int = multiprocessing.cpu_count()-4
    num_classes: int = 10
    epochs: int = 125
    device: str = "cuda"  # "cpu" or "cuda"




    def __post_init__(self):
        ...

def args_to_config(args: Namespace):
    """From the args namespace, create a Configuration.

    It will change all the fields that have ben added to the args.
    If a field is not added in the args will be ignored.
    Fields in the args that are not in the Config this will be ignored.

    Args:
        args (Namespace): Parsed arguments. 

    Returns:
        Configuration: Configuration with args values.
    """
    fields = {f.name for f in dataclasses.fields(Configuration)}
    filtered = {k: v for k, v in vars(args).items() if k in fields}
    return Configuration(**filtered)
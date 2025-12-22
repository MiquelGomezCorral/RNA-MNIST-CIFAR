"""Main file for scripts with arguments and call other functions."""

import argparse
import dotenv
from maikol_utils.other_utils import args_to_dataclass
from src.config import Configuration

from scripts import train_mnist, train_cifar

def cmd_mnist(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    train_mnist(CONFIG)

def cmd_cifar(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    train_cifar(CONFIG)


# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       mnist
    # ======================================================================================
    p_mnist = subparsers.add_parser("mnist", help="Read and extract from config list")
    p_mnist.add_argument(
        "-b", "--batch_size", type=int, default=1024, help="Batch size (default: 1024)", 
    ) 
    p_mnist.add_argument(
        "-lr", "--lr", type=float, default=0.002, help="Learning rate (default: 0.002)",
    )
    p_mnist.add_argument(
        "-wd", "--weight_decay", type=float, default=1e-6, help="Weight decay (default: 1e-6)",
    )
    p_mnist.add_argument(
        "-e", "--epochs", type=int, default=200, help="Number of epochs (default: 200)",
    )
    p_mnist.add_argument(
        "-des", "--description", type=str, default="MNIST classification with MLP using Pytorch", help="Description of the experiment"
    )
    p_mnist.set_defaults(func=cmd_mnist)

    # ======================================================================================
    #                                       cifar
    # ======================================================================================
    p_cifar = subparsers.add_parser("cifar", help="Read and extract from config list")
    p_cifar.add_argument(
        "-b", "--batch_size", type=int, default=1024, help="Batch size (default: 1024)", 
    ) 
    p_cifar.add_argument(
        "-lr", "--lr", type=float, default=0.0025, help="Learning rate (default: 0.0025)",
    )
    p_cifar.add_argument(
        "-wd", "--weight_decay", type=float, default=1e-6, help="Weight decay (default: 1e-6)",
    )
    p_cifar.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of epochs (default: 100)",
    )
    p_cifar.add_argument(
        "-des", "--description", type=str, default="CIFAR classification with CNN using Pytorch", help="Description of the experiment"
    )
    p_cifar.set_defaults(func=cmd_cifar)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)
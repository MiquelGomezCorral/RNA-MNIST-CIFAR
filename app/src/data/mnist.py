import torchvision
import torch
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from maikol_utils.print_utils import print_separator

from src.config import Configuration



class MNIST_dataset(Dataset):
    def __init__(self, data, partition = "train"):
        print("\nLoading MNIST ", partition, " Dataset...")
        self.data = data
        self.partition = partition
        print("\tTotal Len.: ", len(self.data), "\n", 50*"-")

    def __len__(self):
        return len(self.data)
    
    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image_tensor = self.from_pil_to_tensor(image)

        # NOTE: net expect a 784 size vector and our dataset 
        # provide 1x28x28 (channels, height, width) -> 784
        image_tensor = image_tensor.view(-1)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}



def load_mnist(CONFIG: Configuration):
    # Load dataset base
    train_set = torchvision.datasets.MNIST(CONFIG.DATA_FOLDER, train=True, download=True)
    test_set = torchvision.datasets.MNIST(CONFIG.DATA_FOLDER, train=False, download=True)

    # Load dataset MNIST_dataset class
    train_dataset = MNIST_dataset(train_set, partition="train")
    test_dataset = MNIST_dataset(test_set, partition="test")

    # DataLoader Class
    train_dataloader = DataLoader(train_dataset, CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_dataloader = DataLoader(test_dataset, CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    print_separator("MNIST Dataset Loaded")
    print(" - Train images: ", len(train_set))  
    print(" - Test images: ", len(test_set))

    return train_dataloader, test_dataloader


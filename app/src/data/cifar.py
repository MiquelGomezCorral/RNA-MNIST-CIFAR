import torchvision
import torch
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from maikol_utils.print_utils import print_separator

from src.config import Configuration


class CIFAR10_dataset(Dataset):
    def __init__(self, partition = "train", transform = None, CONFIG: Configuration = None):
        self.partition = partition
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])
        self.config = CONFIG
        if self.partition == "train":
            self.data = torchvision.datasets.CIFAR10(
                CONFIG.DATA_FOLDER, 
                train=True,
                download=True
            )
        else:
            self.data = torchvision.datasets.CIFAR10(
                CONFIG.DATA_FOLDER, 
                train=False,
                download=True
            )
        print(f" - Total data {self.partition}: {len(self.data)}")
        print_separator(" ")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]

        # data augmentation
        image_tensor = self.transform(image)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}
    


def load_cifar(CONFIG: Configuration):
    print_separator(f"Loading CIFAR10 Dataset...")
    
    train_da = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomCrop(32, padding=4),
        ], p=CONFIG.aug_prob),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0),
        ], p=CONFIG.aug_prob),
        transforms.RandomApply([
            torchvision.transforms.RandAugment(num_ops=2, magnitude=9),
        ], p=CONFIG.aug_prob),
        transforms.ToTensor(),
    ])

    # # Enhanced data augmentation for training
    # train_da = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.RandAugment(num_ops=2, magnitude=9),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar_mean, std=cifar_std),
    # ])
    
    # Test transform with normalization only
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = CIFAR10_dataset(partition="train", transform=train_da, CONFIG=CONFIG)
    test_dataset = CIFAR10_dataset(partition="test", transform=test_transform, CONFIG=CONFIG)

    # DataLoader Class
    train_dataloader = DataLoader(train_dataset, CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_dataloader = DataLoader(test_dataset, CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    print(" - Train images: ", len(train_dataset))  
    print(" - Test images: ", len(test_dataset))

    return train_dataloader, test_dataloader

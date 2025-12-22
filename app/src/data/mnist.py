import torchvision
import torch
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from maikol_utils.print_utils import print_separator

from src.config import Configuration


class MNIST_dataset(Dataset):
    def __init__(self, data, partition = "train", CONFIG: Configuration = None):
        print_separator(f"Loading MNIST {partition} Dataset...")
        self.data = data
        self.partition = partition
        self.CONFIG = CONFIG
        print(f"Total data {len(self.data)}")
        print_separator(" ")

    def __len__(self):
        return len(self.data)
    
    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image_tensor = self.from_pil_to_tensor(image)

        # data augmentation
        image_tensor = self.data_augmentation(image_tensor, self.partition)

        # NOTE: net expect a 784 size vector and our dataset 
        # provide 1x28x28 (channels, height, width) -> 784
        image_tensor = image_tensor.view(-1)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"idx": idx, "img": image_tensor, "label": label}
        # return {"img": image_tensor, "label": label}

        
    def data_augmentation(self, image_tensor, partition):
        # Apply augmentation only to trainwith 50% probability
        if partition == "test" or torch.rand(1).item() > self.CONFIG.aug_prob: return image_tensor
    
        angle = torch.FloatTensor(1).uniform_(-self.CONFIG.angle_range, self.CONFIG.angle_range).item()
        
        translate_x = torch.FloatTensor(1).uniform_(-self.CONFIG.translate_range, self.CONFIG.translate_range).item() * self.CONFIG.image_size  
        translate_y = torch.FloatTensor(1).uniform_(-self.CONFIG.translate_range, self.CONFIG.translate_range).item() * self.CONFIG.image_size  
        
        scale = torch.FloatTensor(1).uniform_(self.CONFIG.scale_range[0], self.CONFIG.scale_range[1]).item()
        
        shear = torch.FloatTensor(1).uniform_(-self.CONFIG.shear_range, self.CONFIG.shear_range).item()
        
        image_tensor = torchvision.transforms.functional.affine(
            image_tensor,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=[shear, 0],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )

        return image_tensor



def load_mnist(CONFIG: Configuration):
    # Load dataset base
    train_set = torchvision.datasets.MNIST(CONFIG.DATA_FOLDER, train=True, download=True)
    test_set = torchvision.datasets.MNIST(CONFIG.DATA_FOLDER, train=False, download=True)

    # Load dataset MNIST_dataset class
    train_dataset = MNIST_dataset(train_set, partition="train", CONFIG=CONFIG)
    test_dataset = MNIST_dataset(test_set, partition="test", CONFIG=CONFIG)

    # DataLoader Class
    train_dataloader = DataLoader(train_dataset, CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_dataloader = DataLoader(test_dataset, CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    print_separator("MNIST Dataset Loaded")
    print(" - Train images: ", len(train_set))  
    print(" - Test images: ", len(test_set))

    return train_dataloader, test_dataloader

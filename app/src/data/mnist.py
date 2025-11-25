import torchvision
import torch
import torch.nn.functional as  F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from maikol_utils.print_utils import print_separator

from src.config import Configuration



class MNIST_dataset(Dataset):
    def __init__(self, data, partition = "train", CONFIG: Configuration = None):
        print("\nLoading MNIST ", partition, " Dataset...")
        self.data = data
        self.partition = partition
        self.CONFIG = CONFIG
        print("\tTotal Len.: ", len(self.data), "\n", 50*"-")

    def __len__(self):
        return len(self.data)
    
    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image_tensor = self.from_pil_to_tensor(image)

        # data augmentation
        image_tensor = data_augmentation(image_tensor, self.partition, self.CONFIG)

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
    train_dataset = MNIST_dataset(train_set, partition="train", CONFIG=CONFIG)
    test_dataset = MNIST_dataset(test_set, partition="test", CONFIG=CONFIG)

    # DataLoader Class
    train_dataloader = DataLoader(train_dataset, CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_dataloader = DataLoader(test_dataset, CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    print_separator("MNIST Dataset Loaded")
    print(" - Train images: ", len(train_set))  
    print(" - Test images: ", len(test_set))

    return train_dataloader, test_dataloader


# def data_augmentation(image_tensor, partition):
#     if partition == "test": return image_tensor

#     r = torch.rand(1).item()
#     if r > 0.5:
#         return image_tensor  # No augmentation
    
#     r = torch.rand(1).item()
#     options = 3
#     if r < 1 / options:
#         # Add random noise
#         noise = torch.randn(image_tensor.size()) * 0.1
#         image_tensor += noise
#     # elif r < 2/options:
#     #     # Random brightness adjustment
#     #     factor = torch.rand(1).item() * 0.2 + 0.9  # Factor between 0.9 and 1.1
#     #     image_tensor *= factor
#     elif r < 2/options:
#         # Random contrast adjustment
#         factor = torch.rand(1).item() * 0.2 + 0.9  # Factor between 0.9 and 1.1
#         mean = torch.mean(image_tensor)
#         image_tensor = (image_tensor - mean) * factor + mean
#     elif r < 3/options:
#         # Random rotation
#         angles = [-15, -10, -5, 5, 10, 15]
#         angle = angles[torch.randint(0, len(angles), (1,)).item()]
#         image_tensor = torchvision.transforms.functional.rotate(image_tensor, angle)
#     elif r < 4/options:
#         # using affine rotation

#     image_tensor = torch.clamp(image_tensor, 0., 1.)
#     return image_tensor


def data_augmentation(image_tensor, partition, CONFIG: Configuration):
    """
    Apply data augmentation to MNIST images.
    Equivalent to transforms.RandomAffine + Normalize
    """
    # MNIST normalization constants
    
    
    if partition == "train":
        # Apply augmentation with 50% probability
        if torch.rand(1).item() > CONFIG.aug_prob:
            angle = torch.FloatTensor(1).uniform_(-CONFIG.angle_range, CONFIG.angle_range).item()
            translate_x = torch.FloatTensor(1).uniform_(-CONFIG.translate_range, CONFIG.translate_range).item() * CONFIG.image_size  # 10% of image width
            translate_y = torch.FloatTensor(1).uniform_(-CONFIG.translate_range, CONFIG.translate_range).item() * CONFIG.image_size  # 10% of image height
            scale = torch.FloatTensor(1).uniform_(CONFIG.scale_range[0], CONFIG.scale_range[1]).item()
            shear = torch.FloatTensor(1).uniform_(-CONFIG.shear_range, CONFIG.shear_range).item()
            
            image_tensor = torchvision.transforms.functional.affine(
                image_tensor,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[shear, 0],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
    

    return image_tensor
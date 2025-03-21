import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.io import ImageReadMode, read_image


class DeblurringDataset(Dataset):
    def __init__(self, images_path, input_transform=None, target_transform=None, blur_kernel=3):
        self.images_path = images_path
        self.images = os.listdir(images_path)
        self.input_transform = input_transform
        self.target_transform = target_transform
        if not input_transform:
            self.input_transform = transforms.Compose([
                transforms.ConvertImageDtype(dtype=torch.float),
                transforms.Resize((256, 256)),
                v2.GaussianBlur((blur_kernel, blur_kernel), sigma=(1.5,2.5)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if not target_transform:
            self.target_transform = transforms.Compose([
                transforms.ConvertImageDtype(dtype=torch.float),
                transforms.Resize((256, 256)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input = self.input_transform(read_image(os.path.join(self.images_path, self.images[idx]), mode=ImageReadMode.RGB))
        target = self.target_transform(read_image(os.path.join(self.images_path, self.images[idx]), mode=ImageReadMode.RGB))

        return {
            "input": input,
            "target": target
        }

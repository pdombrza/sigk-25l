from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image


class ImageDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = read_image(item["image"], mode=ImageReadMode.RGB)
        label = item["label"]

        if self.transforms:
            image = self.transforms(image)

        return image, label
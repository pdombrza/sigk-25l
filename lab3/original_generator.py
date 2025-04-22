import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import BloodMNIST
from torchvision.utils import save_image
import os

output_dir = "data/bloodmnist_images"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = BloodMNIST(split="train", download=True, transform=transform, size=64)

for idx, (img, label) in enumerate(dataset):
    filename = f"{output_dir}/img_{idx:05d}_label_{label}.png"
    save_image(img, filename)


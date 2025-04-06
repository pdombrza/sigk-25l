import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchmetrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import lpips

class NoisyDataset(Dataset):
    def __init__(self, root_folder, transform=None, sigma=0.02):
        self.transform = transform
        self.sigma = sigma

        self.images = []
        
        for fname in os.listdir(root_folder):
            if fname.endswith((".jpg", ".png")):
                image = Image.open(os.path.join(root_folder, fname)).convert("RGB")
                self.images.append(np.array(image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])

        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = NoisyDataset(root_folder="./data/mantiuk_output", transform=transform, sigma=0.01)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        self.residual_match = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_match(x)
        return self.layers(x) + residual

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)
autoencoder = Autoencoder().to(device)

ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
mse = nn.MSELoss().to(device)
# MSE + SSIM
def combined_loss(output, target, alpha=0.5, beta=0.5):
    mse_loss = mse(output, target)
    ssim_loss = 1 - ssim(output, target) 
    return alpha * mse_loss + beta * ssim_loss

optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Initialize LPIPS loss function (AlexNet backbone)

# Training loop with metrics computation
num_epochs = 50
losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    autoencoder.train()

    for images in dataloader:
        images = images.to(device)

        outputs = autoencoder(images)
        loss = combined_loss(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset
train_dataset = BloodMNIST(split="train", download=True, transform=transform, size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),  # (N, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),         # (N, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),          # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),           # (N, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),            # (N, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # (N, 32, 32, 32)
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1), # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),# (N, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 8),       # (N, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1)

# Initialize models
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss & optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
epochs = 50
for epoch in range(epochs):
    g_loss_total = 0
    d_loss_total = 0
    for imgs, _ in train_loader:
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        valid = torch.ones(batch_size, device=device)
        fake = torch.zeros(batch_size, device=device)

        # Train Generator
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Generator Loss: {g_loss_total/len(train_loader):.4f} | Discriminator Loss: {d_loss_total/len(train_loader):.4f}")

# Generate and show 10 samples from generator
generator.eval()
with torch.no_grad():
    z = torch.randn(10, latent_dim, 1, 1, device=device)
    gen_imgs = generator(z).cpu() * 0.5 + 0.5  # Denormalize

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(np.transpose(gen_imgs[i], (1, 2, 0)))
    axes[i].axis('off')
plt.suptitle("Generated Images from GAN")
plt.tight_layout()
plt.show()

# Show 10 original samples
val_dataset = BloodMNIST(split="val", download=True, transform=transform, size=64)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
original_samples = next(iter(val_loader))[0] * 0.5 + 0.5  # Denormalize

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(np.transpose(original_samples[i], (1, 2, 0)))
    axes[i].axis('off')
plt.suptitle("Original BloodMNIST Samples")
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = BloodMNIST(split="train", download=True, transform=transform, size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class VAE(nn.Module):
    def __init__(self, latent_dim=64, num_classes=8):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
        )

        self.flattened_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def encode(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z).view(z.size(0), 256, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        pred_class = self.classifier(z)
        return recon, mu, logvar, pred_class

# VAE loss (reconstruction + KLD)
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

vae = VAE(latent_dim=32).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
cls_criterion = nn.CrossEntropyLoss()
epochs = 50

vae.train()
for epoch in range(epochs):
    total_loss = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        recon_imgs, mu, logvar, pred_class = vae(imgs)

        recon_kld_loss = vae_loss(recon_imgs, imgs, mu, logvar)
        cls_loss = cls_criterion(pred_class, labels)
        loss = recon_kld_loss + cls_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

# Generate images from latent space
vae.eval()
with torch.no_grad():
    z = torch.randn(10, vae.latent_dim).to(device)
    samples = vae.decode(z).cpu()
    pred_classes = vae.classifier(z).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(np.transpose(samples[i], (1, 2, 0)))
        axes[i].set_title(f"Pred: {pred_classes[i]}")
        axes[i].axis('off')
    plt.suptitle("Generated VAE Samples with Predicted Classes")
    plt.tight_layout()
    plt.show()

# Show real samples
val_dataset = BloodMNIST(split="val", download=True, transform=transform, size=64)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
original_samples = next(iter(val_loader))[0]

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(np.transpose(original_samples[i], (1, 2, 0)))
    axes[i].axis('off')
plt.suptitle("Original Samples from BloodMNIST (Validation Set)")
plt.tight_layout()
plt.show()

# Save large batch of generated images
output_dir = "data/generated_vae"
os.makedirs(output_dir, exist_ok=True)

vae.eval()
num_samples = 12000

with torch.no_grad():
    z = torch.randn(num_samples, vae.latent_dim).to(device)
    gen_imgs = vae.decode(z).cpu()
    preds = vae.classifier(z).argmax(dim=1).cpu().numpy()

    for i in range(num_samples):
        filename = f"sample_{i+1:05d}_class_{preds[i]}.png"
        save_path = os.path.join(output_dir, filename)
        save_image(gen_imgs[i], save_path)

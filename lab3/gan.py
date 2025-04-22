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
    transforms.Normalize([0.5], [0.5])
])

train_dataset = BloodMNIST(split="train", download=True, transform=transform, size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
num_classes = 8

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=8):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),  # (N, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),         # (N, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),          # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),           # (N, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),            # (N, 3, 64, 64)
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, z):
        img = self.gen(z)
        class_logits = self.classifier(z.view(z.size(0), -1))
        return img, class_logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 1, 8),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1)

latent_dim = 100
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator().to(device)

criterion_gan = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
for epoch in range(epochs):
    g_loss_total = 0
    d_loss_total = 0
    for imgs, _ in train_loader:
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, device=device)
        fake = torch.zeros(batch_size, device=device)

        # -------------------
        # Train Generator
        # -------------------
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        class_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        gen_imgs, class_logits = generator(z)
        g_loss_adv = criterion_gan(discriminator(gen_imgs), valid)
        g_loss_cls = criterion_cls(class_logits, class_labels)
        g_loss = g_loss_adv + g_loss_cls

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # -------------------
        # Train Discriminator
        # -------------------
        real_loss = criterion_gan(discriminator(real_imgs), valid)
        fake_loss = criterion_gan(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Generator Loss: {g_loss_total/len(train_loader):.4f} | Discriminator Loss: {d_loss_total/len(train_loader):.4f}")

# Generate and show 10 samples
generator.eval()
with torch.no_grad():
    z = torch.randn(10, latent_dim, 1, 1, device=device)
    gen_imgs, class_logits = generator(z)
    gen_imgs = gen_imgs.cpu() * 0.5 + 0.5
    predicted_classes = class_logits.argmax(dim=1).cpu().numpy()

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(np.transpose(gen_imgs[i], (1, 2, 0)))
    axes[i].set_title(f"Pred: {predicted_classes[i]}")
    axes[i].axis('off')
plt.suptitle("Generated Images with Predicted Classes")
plt.tight_layout()
plt.show()

# Show original images
val_dataset = BloodMNIST(split="val", download=True, transform=transform, size=64)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
original_samples = next(iter(val_loader))[0] * 0.5 + 0.5

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(np.transpose(original_samples[i], (1, 2, 0)))
    axes[i].axis('off')
plt.suptitle("Original BloodMNIST Samples")
plt.tight_layout()
plt.show()

output_dir = "data/generated_gan"
os.makedirs(output_dir, exist_ok=True)

generator.eval()
num_samples = 12000
with torch.no_grad():
    z = torch.randn(num_samples, latent_dim, 1, 1).to(device)
    gen_imgs, class_logits = generator(z)
    gen_imgs = gen_imgs.cpu() * 0.5 + 0.5
    predicted_classes = class_logits.argmax(dim=1).cpu().numpy()

    for i in range(num_samples):
        filename = f"sample_{i+1:05d}_class_{predicted_classes[i]}.png"
        save_path = os.path.join(output_dir, filename)
        save_image(gen_imgs[i], save_path)

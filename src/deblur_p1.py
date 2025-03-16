import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchmetrics
from PIL import Image
import numpy as np
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_bilateral
from image_operations import resize_stretch

def add_gaussian_noise(image, sigma=0.01):
    image_np = np.array(image) / 255.0
    noisy_image = random_noise(image_np, mode='gaussian', var=sigma**2)
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

class NoisyDataset(Dataset):
    def __init__(self, root_folder, transform=None, resize_func=resize_stretch, sigma=0.02):
        self.transform = transform
        self.sigma = sigma
        self.resize_func = resize_func if resize_func else (lambda x: x)

        self.images = []
        self.noisy_images = []
        
        for fname in os.listdir(root_folder):
            if fname.endswith((".jpg", ".png")):
                image = Image.open(os.path.join(root_folder, fname)).convert("RGB")
                image = self.resize_func(image)
                image_np = np.array(image)
                noisy_image = add_gaussian_noise(image, sigma=self.sigma)
                self.images.append(image_np)
                self.noisy_images.append(np.array(noisy_image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        noisy_image = Image.fromarray(self.noisy_images[idx])

        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)

        return noisy_image, image

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = NoisyDataset(root_folder="data/DIV2K_train_LR_mild", transform=transform, sigma=0.1)
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
autoencoder = Autoencoder().to(device)

ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
mse = nn.MSELoss().to(device)
# MSE + SSIM
def combined_loss(output, target, alpha=0.5, beta=0.5):
    mse_loss = mse(output, target)
    ssim_loss = 1 - ssim(output, target) 
    return alpha * mse_loss + beta * ssim_loss

optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 50
losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_imgs, clean_imgs in dataloader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        outputs = autoencoder(noisy_imgs)
        loss = combined_loss(outputs, clean_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    losses.append(running_loss/len(dataloader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

print("Training complete!")

def compare_with_bilateral(noisy_img, clean_img, autoencoder):
    autoencoder.eval()
    with torch.no_grad():
        noisy_tensor = transform(noisy_img).unsqueeze(0).to(device)
        autoencoder_output = autoencoder(noisy_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    noisy_np = np.array(noisy_img) / 255.0
    bilateral_output = denoise_bilateral(noisy_np, sigma_color=0.02, sigma_spatial=5, channel_axis=-1)

    autoencoder_output = np.clip(autoencoder_output, 0, 1)
    bilateral_output = np.clip(bilateral_output, 0, 1)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 2 columns

    # Top row
    ax[0, 0].imshow(clean_img)
    ax[0, 0].set_title("Oryginalny obraz")

    ax[0, 1].imshow(autoencoder_output)
    ax[0, 1].set_title("Wyjście autoenkodera")

    ax[1, 0].imshow(bilateral_output)
    ax[1, 0].set_title("Wyjście filtra dwustronnego")

    ax[1, 1].imshow(noisy_img)
    ax[1, 1].set_title("Obraz zaszumiony")


    for a in ax.ravel():
        a.axis("off")
    plt.show()

def plot():
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epoka')
    plt.ylabel('MSE + SSIM')
    plt.title('Postęp uczenia modelu')
    plt.legend()
    plt.grid(True)
    plt.show()

noisy_sample, clean_sample = dataset[40]
compare_with_bilateral(transforms.ToPILImage()(noisy_sample), transforms.ToPILImage()(clean_sample), autoencoder)
noisy_sample, clean_sample = dataset[101]
compare_with_bilateral(transforms.ToPILImage()(noisy_sample), transforms.ToPILImage()(clean_sample), autoencoder)
noisy_sample, clean_sample = dataset[202]
compare_with_bilateral(transforms.ToPILImage()(noisy_sample), transforms.ToPILImage()(clean_sample), autoencoder)
noisy_sample, clean_sample = dataset[203]
compare_with_bilateral(transforms.ToPILImage()(noisy_sample), transforms.ToPILImage()(clean_sample), autoencoder)
plot()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from PIL import Image
from torch.utils.data import Dataset

# Config
num_classes = 8
batch_size = 64
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Custom dataset loader using filename to get class
def get_custom_dataset(path):
    class CustomDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
            label = int(self.image_files[idx].split('_')[3].split('.')[0])  # Convert it to integer
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

        def __len__(self):
            return len(self.image_files)

    return CustomDataset(path, transform=transform)

# Model
class MedModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, test_loader):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds))

# Setup each scenario
def run_experiment(train_path, test_path, name):
    print(f"\n--- Running experiment: {name} ---")
    train_dataset = get_custom_dataset(train_path)
    test_dataset = get_custom_dataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = MedModel(num_labels=num_classes)
    train_model(model, train_loader, test_loader)

# Experiments
run_experiment("data/bloodmnist_images", "data/generated_gan", "Train on real, test on GAN")
run_experiment("data/generated_gan", "data/bloodmnist_images", "Train on GAN, test on real")
run_experiment("data/mixed_gan", "data/mixed_gan", "Train & test on mixed GAN")

run_experiment("data/bloodmnist_images", "data/generated_vae", "Train on real, test on VAE")
run_experiment("data/generated_vae", "data/bloodmnist_images", "Train on VAE, test on real")
run_experiment("data/mixed_vae", "data/mixed_vae", "Train & test on mixed VAE")
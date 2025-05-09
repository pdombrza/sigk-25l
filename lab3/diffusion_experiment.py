import torch
import torchvision
from classifier import MedModel
from torchvision import transforms
from medmnist import BloodMNIST
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data import Dataset

class BloodMNISTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label = int(label)
        return image, label

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    medmnist_train = BloodMNISTWrapper(BloodMNIST(split="train", download=True, transform=transform, size=64))
    medmnist_test = BloodMNISTWrapper(BloodMNIST(split="test", download=True, transform=transform, size=64))

    num_classes = 8
    num_samples_per_class = 100
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(medmnist_train):
        if len(class_indices[label]) < num_samples_per_class:
            class_indices[label].append(idx)
        if all(len(indices) == num_samples_per_class for indices in class_indices.values()):
            break

    selected_indices = [idx for indices in class_indices.values() for idx in indices]
    bloodmnist_subset = Subset(medmnist_train, selected_indices)

    generated_data = torchvision.datasets.ImageFolder("lab3/examples", transform=transform)

    mixed_dataset = ConcatDataset([bloodmnist_subset, generated_data])
    train_mixed, test_mixed = torch.utils.data.random_split(mixed_dataset, [len(mixed_dataset) - len(generated_data), len(generated_data)])

    print("Running Experiment 1: Train and test on mixed data")
    run_experiment(train_mixed, test_mixed, "Train and test on mixed data")

    print("Running Experiment 2: Train on BloodMNIST and test on generated data")
    run_experiment(medmnist_train, generated_data, "Train on real, test on generated")

    print("Running Experiment 3: Train on generated data and test on BloodMNIST")
    run_experiment(generated_data, medmnist_test, "Train on generated, test on real")


def run_experiment(train_dataset, test_dataset, experiment_name):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = MedModel(num_labels=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

    print(f"Testing {experiment_name}...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 8
    class_total = [0] * 8

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    accuracy = 100 * correct / total
    print(f"Overall Accuracy for {experiment_name}: {accuracy:.2f}%")

    for i in range(8):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Class {i} Accuracy: {class_accuracy:.2f}%")
        else:
            print(f"Class {i} Accuracy: No samples")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy for {experiment_name}: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from model import EnhancedDetailNet
from utils.metrics import accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_dataset = CustomDataset(...)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create the model
model = EnhancedDetailNet(...)
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Evaluation loop
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    print(f"Accuracy: {accuracy}%")
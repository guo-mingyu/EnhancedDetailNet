import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset

from model import EnhancedDetailNet
from utils.metrics import accuracy
from torchvision.transforms import ToTensor, Resize, Compose


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of classes and input channels
num_classes = 15
input_channels = 3

# Load the test dataset
test_dataset = CustomDataset("./test.txt", input_size=(32, 32), transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create the model
model = EnhancedDetailNet(num_classes=num_classes, input_channels=input_channels)
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load("model_epoch10.pth"))
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
    print(f"Accuracy: {accuracy:.2f}%")

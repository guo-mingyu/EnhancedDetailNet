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
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Create the model
model = EnhancedDetailNet(num_classes=num_classes, input_channels=input_channels)
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load("model_epoch300300.pth"))
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

        # Print the prediction for each image
        for i in range(images.size(0)):
            print(f"Image {i + 1}: Predicted class {predicted[i]}, lables {labels[i]}, outputs.data {outputs.data}")

    accuracy = 100 * total_correct / total_samples
    print(f"Accuracy: {accuracy:.2f}%")

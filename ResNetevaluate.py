import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
from utils.dataset import CustomDataset
import argparse

# Define the ResNet model
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--model_path", type=str, default="model_epoch_Resnet50_lr1e-6_100.pth", help="Path to the trained model")
parser.add_argument("--input_size", type=int, default=32, help="Input image size")
args = parser.parse_args()

# Load the test dataset
test_dataset = CustomDataset("./test.txt", input_size=(args.input_size, args.input_size), transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Create the ResNet-50 model
model = ResNetModel(num_classes=args.num_classes)
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load(args.model_path))
model.eval()

print(model)

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

# Get the memory allocated in megabytes (MB)
memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
print(f"Memory Allocated: {memory_allocated:.2f} MB")

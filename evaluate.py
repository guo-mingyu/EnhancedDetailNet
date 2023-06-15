import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import EnhancedDetailNet
from utils.dataset import CustomDataset
from utils.metrics import accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_dataset = CustomDataset('data/test/')
test_loader = DataLoader(test_dataset)

# Load the trained model
model = EnhancedDetailNet().to(device)
model.load_state_dict(torch.load('checkpoints/model.pth'))
model.eval()

# Evaluation loop
test_acc = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute accuracy
        acc = accuracy(outputs, labels)

        test_acc += acc * images.size(0)

# Compute average accuracy
test_acc /= len(test_dataset)

print(f"Test Accuracy: {test_acc:.4f}")

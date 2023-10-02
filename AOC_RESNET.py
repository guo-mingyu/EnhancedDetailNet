import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset

from model import EnhancedDetailNet
from utils.metrics import accuracy
from torchvision.transforms import ToTensor, Resize, Compose
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of classes and input channels
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
parser.add_argument("--channel_mode", type=str, default="normal",
                    help="Channel mode: lightweight, mode: normal, normal, advanced")
parser.add_argument("--model_path", type=str, default="model_epoch_lr1e-05_advanced_100.pth",
                    help="Path to the trained model")
args = parser.parse_args()

print(args)

# Load the test dataset
test_dataset = CustomDataset("./test.txt", input_size=(32, 32), transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Create the model
model = EnhancedDetailNet(num_classes=args.num_classes, input_channels=args.input_channels,
                          channel_mode=args.channel_mode)
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load(args.model_path))
model.eval()

print(model)

# Lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Convert predicted class to class name
class_names = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
               "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
               "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
               "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]


# Evaluation loop
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(probabilities.cpu().numpy())

    # Convert lists to numpy arrays
    true_labels_np = np.array(true_labels)
    predicted_probs_np = np.array(predicted_probs)

    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(20, 20))
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_np == i, predicted_probs_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    # plt.figure()
    for i in range(args.num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {class_names[i]}(AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # Save the figure
    plt.savefig(f'ROC_curves{args.model_path}.png')

    accuracy = 100 * total_correct / total_samples
    print(f"Accuracy: {accuracy:.2f}%")

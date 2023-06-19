import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from model import EnhancedDetailNet

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving the model")
args = parser.parse_args()

# Create an instance of the model
model = EnhancedDetailNet(num_classes=args.num_classes, input_channels=args.input_channels)

# Print the model summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(args.input_channels, 32, 32))

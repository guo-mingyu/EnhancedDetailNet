import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from model import EnhancedDetailNet
from torchvision.models import resnet50

# Define the ResNet model
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--channel_mode", type=str, default="advanced-300",
                    help="Channel mode: lightweight, mode: normal, normal, advanced")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving the model")
args = parser.parse_args()

# Create an instance of the model
model = ResNetModel(num_classes=args.num_classes)

# Print the model summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(args.input_channels, 32, 32))

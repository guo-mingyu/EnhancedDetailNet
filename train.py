import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import argparse

from model import EnhancedDetailNet
from utils.metrics import accuracy
from utils.dataset import CustomDataset

from torchvision.transforms import ToTensor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving the model")
args = parser.parse_args()

# Load the dataset
train_dataset = CustomDataset("./train.txt", input_size=(32, 32), transform=ToTensor())
val_dataset = CustomDataset("./val.txt", input_size=(32, 32), transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# Create the model
model = EnhancedDetailNet(num_classes=args.num_classes, input_channels=args.input_channels)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(args.num_epochs):
    model.train()  # Set the model to train mode
    epoch_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()  # 记录每个epoch的开始时间
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training loss
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            #print(f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")
            batch_accuracy = 100 * total_correct / total_samples
            batch_error_rate = 100 - batch_accuracy
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}, Accuracy: {batch_accuracy}%, Error Rate: {batch_error_rate}%")

    epoch_loss /= len(train_loader)
    epoch_accuracy = 100 * total_correct / total_samples
    end_time = time.time()  # 记录每个epoch的结束时间
    epoch_time = end_time - start_time  # 计算每个epoch的持续时间
    print(f"Epoch [{epoch + 1}], Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy}%, Time: {epoch_time}s")

    # Validation
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        val_start_time = time.time()  # 记录验证过程的开始时间
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_samples += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                val_batch_accuracy = 100 * val_correct / val_samples
                val_batch_error_rate = 100 - val_batch_accuracy
                print(f"Validation Step [{i + 1}/{len(val_loader)}], Loss: {loss.item()}, Accuracy: {val_batch_accuracy}%, Error Rate: {val_batch_error_rate}%")

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_samples
        val_end_time = time.time()  # 记录验证过程的结束时间
        val_time = val_end_time - val_start_time  # 计算验证过程的持续时间

        print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}%, Time: {val_time}s")

    # Print average accuracy and loss
    avg_train_loss = epoch_loss
    avg_train_accuracy = epoch_accuracy
    avg_val_loss = val_loss
    avg_val_accuracy = val_accuracy

    print(f"Average Training Loss: {avg_train_loss}, Average Training Accuracy: {avg_train_accuracy}%, Average Validation Loss: {avg_val_loss}, Average Validation Accuracy: {avg_val_accuracy}%")

    # Save the trained model
    if (epoch + 1) % args.save_interval == 0:
        torch.save(model.state_dict(), f"model_epoch{epoch + 1}.pth")

print("Training complete!")

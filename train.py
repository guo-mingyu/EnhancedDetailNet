import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import EnhancedDetailNet
from utils.metrics import accuracy
from torchvision.transforms import ToTensor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

# Hyperparameters
num_classes = 15  # 根据数据集的类别数进行设置
input_channels = 3  # 输入图像的通道数，例如RGB图像为3
batch_size = 1  # 批次大小
learning_rate = 0.001  # 学习率
num_epochs = 100  # 训练轮数


# Load the dataset
train_dataset = CustomDataset("./train.txt", transform=ToTensor())
val_dataset = CustomDataset("./val.txt", transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create the model
model = EnhancedDetailNet(num_classes=num_classes, input_channels=input_channels)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()  # Set the model to train mode
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
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")

    # Save the trained model after each epoch
    torch.save(model.state_dict(), f"model_epoch{epoch + 1}.pth")

print("Training complete!")

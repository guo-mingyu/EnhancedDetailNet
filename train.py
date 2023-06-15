import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.dataset import CustomDataset
from utils.metrics import accuracy

# 设置超参数
input_shape = (3, 64, 64)  # 输入图像的形状
num_classes = 10  # 类别数量
batch_size = 64  # 批次大小
learning_rate = 0.001  # 学习率
num_epochs = 10  # 训练轮数

# 创建数据转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建训练集和验证集
train_dataset = CustomDataset("data/train/", transform=transform)
val_dataset = CustomDataset("data/val/", transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建模型实例
model = EnhancedDetailNet(input_shape, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练损失和准确率
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

    # 计算训练集的平均损失和准确率
    train_loss = running_loss / len(train_dataset)
    train_accuracy = correct_predictions / len(train_dataset)

    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    val_correct_predictions = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计验证损失和准确率
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct_predictions += (predicted == labels).sum().item()

    # 计算验证集的平均损失和准确率
    val_loss = val_loss / len(val_dataset)
    val_accuracy = val_correct_predictions / len(val_dataset)

    # 打印训练过程中的指标
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

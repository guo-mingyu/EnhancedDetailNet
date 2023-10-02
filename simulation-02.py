import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import os
import numpy as np
#from model import EnhancedDetailNet


class EnhancedDetailNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EnhancedDetailNet, self).__init__()

        # Input layer
        self.input_layer = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Convolutional module
        self.conv_module = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU()
        )

        # Attention module
        self.self_attention = SelfAttentionModule(32)
        self.multi_scale_attention = MultiScaleAttentionModule(32)

        # Pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # Enhanced convolutional layer
        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 减小通道数为16
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Global feature encoding
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # Classification/regression layer
        self.fc = nn.Linear(32, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        feature_maps = []
        print("Input layer:", x.shape)
        x = self.relu(x)
        feature_maps.append(x)

        # Convolutional module
        residual = self.residual(x)
        feature_maps.append(x)
        x = self.conv_module(x)
        x = x + residual
        x = self.relu(x)
        feature_maps.append(x)
        print("Convolutional module:", x.shape)

        # Attention module
        x = self.self_attention(x)
        feature_maps.append(x)
        print("Self-attention module:", x.shape)
        x = self.multi_scale_attention(x)
        feature_maps.append(x)
        print("Multi-scale attention module:", x.shape)

        # Pooling layer
        x = self.pooling(x)
        feature_maps.append(x)
        print("Pooling layer:", x.shape)

        # Enhanced convolutional layer
        x = self.enhanced_conv(x)
        feature_maps.append(x)
        x = self.relu(x)
        feature_maps.append(x)
        print("Enhanced convolutional layer:", x.shape)

        # Global feature encoding
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        feature_maps.append(x)
        print("Global feature encoding:", x.shape)

        # Classification/regression layer
        x = self.fc(x)
        feature_maps.append(x)
        print("Classification/regression layer:", x.shape)

        # Dropout layer
        x = self.dropout(x)
        feature_maps.append(x)
        print("dropout layer:", x.shape)

        # Softmax layer
        x = self.softmax(x)
        feature_maps.append(x)
        print("softmax layer:", x.shape)

        return x, feature_maps

class SelfAttentionModule(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionModule, self).__init__()

        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)

        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)

        value = self.value(x).view(batch_size, channels, height * width)

        out = torch.bmm(attention.permute(0, 2, 1), value.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)

        out = self.gamma * out + x

        return out


class MultiScaleAttentionModule(nn.Module):
    def __init__(self, channels):
        super(MultiScaleAttentionModule, self).__init__()

        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)

        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)

        value = self.value(x).view(batch_size, channels, height * width)

        out = torch.bmm(attention.permute(0, 2, 1), value.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)

        out = self.gamma * out + x

        return out


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of channels and classes
input_channels = 3  # 输入图片的通道数
num_classes = 15  # 分类任务的类别数

# Load the model
model = EnhancedDetailNet(input_channels=input_channels, num_classes=num_classes)
model = model.to(device)

print(model)
# Load the input image
image = Image.open("./data/test.jpg")

# Define the transformations
transform = Compose([
    Resize((128, 128)),
    ToTensor()
])

# Apply transformations to the image
image = transform(image).unsqueeze(0).to(device)

# Forward pass through the model and get the feature maps
outputs, feature_maps = model(image)

# Create a directory to save the feature maps
os.makedirs("feature_maps32", exist_ok=True)

# Save each feature map as a PNG file
for i, fm in enumerate(feature_maps):
    layer_name = None
    if i == 0:
        layer_name = "input_layer"
    elif i == 1:
        layer_name = "conv_module"
    elif i == 2:
        layer_name = "self_attention"
    elif i == 3:
        layer_name = "multi_scale_attention"
    elif i == 4:
        layer_name = "pooling"
    elif i == 5:
        layer_name = "enhanced_conv"
    elif i == 6:
        layer_name = "global_pooling"
    elif i == 7:
        layer_name = "full_connection_fc"
    elif i == 8:
        layer_name = "dropout"
    elif i == 9:
        layer_name = "softmax"
    else:
        layer_name = f"output_{i - 10}"

    fm = fm.squeeze().cpu().detach().numpy()  # 去除冗余的维度并转换为NumPy数组
    print(f"fm.shape: {fm.shape}")
    if len(fm.shape) == 0:
        fm = np.expand_dims(fm, axis=0)
        fm = (fm - np.min(fm)) / (np.max(fm) - np.min(fm) + 1e-8)
        fm = (fm * 255).astype(np.uint8)
        fm_image = Image.fromarray(fm.reshape(-1, 1), mode="L")
        fm_image.save(f"feature_maps32/feature_map_{i}_layer_{layer_name}.png")
        print(f"Feature Map _layer_{layer_name}_{i} saved.")
    elif len(fm.shape) == 1:
        fm = (fm - np.min(fm)) / (np.max(fm) - np.min(fm) + 1e-8)
        fm = (fm * 255).astype(np.uint8)
        fm_image = Image.fromarray(fm.reshape(-1, 1), mode="L")
        fm_image.save(f"feature_maps32/feature_map_{i}_layer_{layer_name}.png")
        print(f"Feature Map _layer_{layer_name}_{i} saved.")
    elif len(fm.shape) == 2:
        # If the feature map is already 2D, save it as grayscale image
        fm = (fm - np.min(fm)) / (np.max(fm) - np.min(fm) + 1e-8)  # 将特征图的值缩放到 0-1 范围内
        fm = (fm * 255).astype(np.uint8)  # 将特征图的值转换为整数类型
        fm_image = Image.fromarray(fm, mode="RGB")  # 创建灰度图像对象
        fm_image.save(f"feature_maps32/feature_map_{i + 1}_layer_{layer_name}.png")
        print(f"Feature Map {i+1}_layer_{layer_name} saved.")
    elif len(fm.shape) == 3:
        # If the feature map has 3 dimensions, save each channel as grayscale image
        for channel in range(fm.shape[0]):
            fm_channel = fm[channel, :, :]
            fm_channel = (fm_channel - np.min(fm_channel)) / (np.max(fm_channel) - np.min(fm_channel) + 1e-8)  # 将特征图的值缩放到 0-1 范围内
            fm_channel = (fm_channel * 255).astype(np.uint8)  # 将特征图的值转换为整数类型
            fm_image = Image.fromarray(fm_channel, mode="L")  # 创建灰度图像对象
            fm_image.save(f"feature_maps32/feature_map_{i + 1}_layer_{layer_name}_channel_{channel + 1}.png")
            print(f"Feature Map_layer_{layer_name}_{i+1} Channel {channel+1} saved.")

print("Feature maps saved successfully.")

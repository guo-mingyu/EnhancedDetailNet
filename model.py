import torch
import torch.nn as nn

class EnhancedDetailNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EnhancedDetailNet, self).__init__()

        # Input layer
        self.input_layer = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Convolutional module
        self.conv_module = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1),
            nn.ReLU()
        )

        # Attention module
        self.self_attention = SelfAttentionModule(8)
        self.multi_scale_attention = MultiScaleAttentionModule(8)

        # Pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # Enhanced convolutional layer
        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),  # 减小通道数为16
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Global feature encoding
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # Classification/regression layer
        self.fc = nn.Linear(32, num_classes)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        #print("Input layer:", x.shape)
        x = self.relu(x)

        # Convolutional module
        residual = self.residual(x)
        x = self.conv_module(x)
        x = x + residual
        x = self.relu(x)
        #print("Convolutional module:", x.shape)

        # Attention module
        x = self.self_attention(x)
        #print("Self-attention module:", x.shape)
        x = self.multi_scale_attention(x)
        #print("Multi-scale attention module:", x.shape)

        # Pooling layer
        x = self.pooling(x)
        #print("Pooling layer:", x.shape)

        # Enhanced convolutional layer
        x = self.enhanced_conv(x)
        x = self.relu(x)
        #print("Enhanced convolutional layer:", x.shape)

        # Global feature encoding
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        #print("Global feature encoding:", x.shape)

        # Classification/regression layer
        x = self.fc(x)
        #print("Classification/regression layer:", x.shape)

        # Dropout layer
        x = self.dropout(x)

        # Softmax layer
        x = self.softmax(x)

        return x

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

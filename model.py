import torch
import torch.nn as nn

class EnhancedDetailNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EnhancedDetailNet, self).__init__()

        # Input layer
        self.input_layer = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Convolutional module
        self.conv_module = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Attention module
        self.self_attention = SelfAttentionModule(32)
        self.multi_scale_attention = MultiScaleAttentionModule(32)

        # Pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # Enhanced convolutional layer
        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Global feature encoding
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # Classification/regression layer
        self.fc = nn.Linear(128, num_classes)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.relu(x)

        # Convolutional module
        residual = self.residual(x)
        x = self.conv_module(x)
        x = x + residual
        x = self.relu(x)

        # Attention module
        x = self.self_attention(x)
        x = self.multi_scale_attention(x)

        # Pooling layer
        x = self.pooling(x)

        # Enhanced convolutional layer
        x = self.enhanced_conv(x)
        x = self.relu(x)

        # Global feature encoding
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)

        # Classification/regression layer
        x = self.fc(x)

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

        value = self.value(x).view(batch_size, -1, height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

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

        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width).permute(0, 2, 1)

        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)

        value = self.value(x).view(batch_size, -1, height * width)

        out = torch.bmm(attention.permute(0, 2, 1), value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x

        return out

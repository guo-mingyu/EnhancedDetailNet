import torch
import torch.nn as nn

class EnhancedDetailNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EnhancedDetailNet, self).__init__()

        # Input layer
        self.input_layer = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Convolutional module
        self.conv_module = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Attention module
        self.self_attention = SelfAttentionModule(64)
        self.multi_scale_attention = MultiScaleAttentionModule(64)

        # Pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # Enhanced convolutional layer
        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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
        #self.dropout = nn.Dropout(0.5)

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
        #x = self.dropout(x)

        # Softmax layer
        x = nn.functional.softmax(x, dim=-1)

        return x


class SelfAttentionModule(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionModule, self).__init__()

        self.query = nn.Conv2d(channels, channels // 16, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 16, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.matmul(query, key)
        attention = nn.functional.softmax(attention, dim=-1)

        out = torch.matmul(attention, value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x

        return out


class MultiScaleAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(MultiScaleAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out * x + max_out * x

        return out

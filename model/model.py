import torch
import torch.nn as nn

class EnhancedDetailNet(nn.Module):
    def __init__(self, input_channels, num_classes, channel_mode):
        super(EnhancedDetailNet, self).__init__()

        if channel_mode == 'lightweight':
            conv_channels = 18
        elif channel_mode == 'normal':
            conv_channels = 50
        elif channel_mode == 'advanced':
            conv_channels = 100
        elif channel_mode == 'advanced-200':
            conv_channels = 200
        elif channel_mode == 'advanced-300':
            conv_channels = 300
        elif channel_mode == 'advanced-600':
            conv_channels = 600
        elif channel_mode == 'advanced-1000':
            conv_channels = 1000
        else:
            raise ValueError("Invalid channel mode. Please choose 'lightweight', 'normal', or 'advanced'.")

        # Input layer
        self.input_layer = nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()

        # Convolutional module
        if channel_mode == "lightweight":
            # Lightweight convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        elif channel_mode == "normal":
            # Normal convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        elif channel_mode == "advanced":
            # Advanced convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        elif channel_mode == "advanced-200":
            # Advanced convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        elif channel_mode == "advanced-300":
            # Advanced convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        elif channel_mode == "advanced-600":
            # Advanced convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        elif channel_mode == "advanced-1000":
            # Advanced convolutional module
            self.conv_module = nn.Sequential(
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        else:
            raise ValueError("Invalid channel mode")

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU()
        )

        # Attention module
        self.self_attention = SelfAttentionModule(conv_channels)
        self.multi_scale_attention = MultiScaleAttentionModule(conv_channels)

        # Pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # Enhanced convolutional layer
        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Global feature encoding
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # Classification/regression layer
        self.fc = nn.Linear(conv_channels * 2, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

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

        self.query = nn.Conv2d(channels, channels // 16, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 16, kernel_size=1)
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

        self.query = nn.Conv2d(channels, channels // 16, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 16, kernel_size=1)
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

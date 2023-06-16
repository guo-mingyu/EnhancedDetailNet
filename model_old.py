import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1x1卷积核计算注意力权重
        attention_1x1 = self.conv(x)
        # 使用sigmoid函数将权重限制在0到1之间
        attention_1x1 = self.sigmoid(attention_1x1)
        # 缩放注意力权重
        scaled_attention = x * attention_1x1

        # 其他尺度的注意力计算
        # ...

        return scaled_attention

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.multi_scale_attention = MultiScaleAttention(in_channels)

    def forward(self, x):
        # 自注意力计算
        self_attention = self.conv(x)
        self_attention = self.sigmoid(self_attention)

        # 多尺度注意力计算
        scaled_attention = self.multi_scale_attention(x)

        # 引入残差连接
        attention_output = x + scaled_attention
        attention_output = attention_output * self_attention

        return attention_output

class EnhancedDetailNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(EnhancedDetailNet, self).__init__()
        channels, height, width = input_shape

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.attention_modules = nn.ModuleList([AttentionModule(64) for _ in range(8)])

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 第一层卷积
        x = F.relu(self.conv1(x))
        # 第二层卷积
        x = F.relu(self.conv2(x))

        # 堆叠多个注意力模块
        for attention_module in self.attention_modules:
            residual = x
            x = F.relu(attention_module(x))
            x = x + residual

        # 全局平均池化
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 输出层使用softmax激活函数
        x = F.softmax(x, dim=1)

        return x

import torch
import torch.nn as nn
from torchviz import make_dot


from torchsummary import summary

class EnhancedDetailNet(nn.Module):
    def __init__(self):
        super(EnhancedDetailNet, self).__init__()

        self.input_layer = nn.Conv2d(3, 300, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv_module = nn.Sequential(
            nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(300, 300, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.self_attention = SelfAttentionModule(channels=300)
        self.multi_scale_attention = MultiScaleAttentionModule(channels=300)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(300, 600, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(600, 15)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        residual = self.residual(x)
        x = self.conv_module(x)
        x = x + residual
        x = self.relu(x)
        x = self.self_attention(x)
        x = self.multi_scale_attention(x)
        x = self.pooling(x)
        x = self.enhanced_conv(x)
        x = self.relu(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
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


# 创建模型实例
model = EnhancedDetailNet()

# 创建示例输入
example_input = torch.randn(1, 3, 32, 32)

# 生成模型计算图
graph = make_dot(model(example_input), params=dict(model.named_parameters()))

# 保存计算图为图片
graph.format = 'png'
graph.render('enhanced_detail_net')


# 选择要可视化的重要节点
important_node1 = model.conv_module
important_node2 = model.self_attention
important_node3 = model.multi_scale_attention

# 创建重要节点的示例输入
input_node1 = torch.randn(1, 300, 32, 32)
input_node2 = torch.randn(1, 300, 32, 32)
input_node3 = torch.randn(1, 300, 32, 32)

# 生成重要节点1的计算图
output1 = important_node1(input_node1)
graph_important1 = make_dot(output1, params=dict(important_node1.named_parameters()))
graph_important1.format = 'png'
graph_important1.render('enhanced_detail_net_important1')

# 生成重要节点2的计算图
output2 = important_node2(input_node1)
graph_important2 = make_dot(output2, params=dict(important_node2.named_parameters()))
graph_important2.format = 'png'
graph_important2.render('enhanced_detail_net_important2')

# 生成重要节点3的计算图
output3 = important_node3(input_node1)
graph_important3 = make_dot(output3, params=dict(important_node3.named_parameters()))
graph_important3.format = 'png'
graph_important3.render('enhanced_detail_net_important3')
# 显示计算图
#graph.view()

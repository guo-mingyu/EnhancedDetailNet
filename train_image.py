import os
from PIL import Image
from data.dataset import PlantVillageDataset

# 定义数据集根目录和目标文件路径
data_root = 'PlantVillage'
train_file = 'train.txt'
val_file = 'val.txt'

# 创建训练数据集实例
train_dataset = PlantVillageDataset(root_dir=data_root, transform=ToTensor())

# 创建验证数据集实例
val_dataset = PlantVillageDataset(root_dir=data_root, transform=ToTensor())

# 创建保存图像名的文件
with open(train_file, 'w') as f:
    for image_path, _ in train_dataset:
        image_name = os.path.basename(image_path)
        f.write(image_name + '\n')

with open(val_file, 'w') as f:
    for image_path, _ in val_dataset:
        image_name = os.path.basename(image_path)
        f.write(image_name + '\n')

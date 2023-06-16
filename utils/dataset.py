import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
from PIL import Image
import glob


class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.transform = transform
        self.data = []

        # Load and preprocess the data
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    image_path, label = line.strip().split(',')
                    image_paths = glob.glob(image_path)
                    for path in image_paths:
                        image = Image.open(path)
                        self.data.append({'image': image, 'label': int(label)})
                except ValueError as e:
                    print(f"Ignoring line: {line.strip()} - Invalid format")

    def __getitem__(self, index):
        image = self.data[index]['image']
        label = self.data[index]['label']

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)

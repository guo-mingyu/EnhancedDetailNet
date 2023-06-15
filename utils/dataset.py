import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []  # Placeholder for data

        # Load and preprocess the data
        # Add your code here to read and preprocess the data

    def __getitem__(self, index):
        image = self.data[index]['image']
        label = self.data[index]['label']

        # Apply transformations
        image = ToTensor()(image)

        return image, label

    def __len__(self):
        return len(self.data)

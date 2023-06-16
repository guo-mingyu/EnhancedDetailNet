import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

    def _load_data(self):
        image_paths = []
        labels = []
        class_names = os.listdir(self.root_dir)

        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            image_names = os.listdir(class_dir)

            for image_name in image_names:
                image_path = os.path.join(class_dir, image_name)
                image_paths.append(image_path)
                labels.append(i)

        return image_paths, labels

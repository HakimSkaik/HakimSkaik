# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os

class Comp472OriginalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.labels = [self.get_label(fname) for fname in os.listdir(root_dir)]

    def get_label(self, filename):
        # Implement logic to extract label from filename or other sources
        label = int(filename.split('_')[1])  # Example: extract label from filename
        return label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UvpDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        relative_path = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('relative_path')]
        img_name = os.path.join(self.root_dir, relative_path)
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('label')]

        if self.transform:
            image = self.transform(image)

        return image, label
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class UvpDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, train=True):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        if self.train:
            # 20% data is used for evaluation
            self.data_frame, _ = train_test_split(self.data_frame, test_size=0.2, random_state=42)

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
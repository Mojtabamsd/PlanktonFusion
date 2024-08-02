import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class UvpDataset(Dataset):
    def __init__(self, root_dir, num_class, csv_file=None, transform=None, phase='train', gray=True,
                 permitted_formats=None):
        self.root_dir = os.path.join(root_dir, 'output')
        if not os.path.exists(self.root_dir):
            self.root_dir = root_dir
        self.num_class = num_class
        self.csv_file = csv_file
        self.transform = transform
        self.phase = phase
        self.gray = gray
        self.permitted_formats = permitted_formats

        if self.csv_file:
            self.data_frame = pd.read_csv(csv_file)

        if self.phase == 'train_val':
            self.data_frame, self.data_frame_val = train_test_split(self.data_frame, test_size=0.2, random_state=42)

        # regrouping
        ren = pd.read_csv("./data_preparation/regrouping.csv")
        if self.num_class == 13:
            self.label_to_int = {label: i for i, label in enumerate(ren['regrouped2'].unique())}
        elif self.num_class == 25:
            self.label_to_int = {label: i for i, label in enumerate(ren['regrouped1'].unique())}
        else:
            unique_labels = sorted(self.data_frame['label'].unique())
            self.label_to_int = {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        if self.csv_file and self.phase == 'val':
            return len(self.data_frame_val)
        elif self.csv_file:
            return len(self.data_frame)
        else:
            return len([filename for filename in os.listdir(self.root_dir) if self.is_permitted_format(filename)])

    def __getitem__(self, idx):
        if self.csv_file and self.phase == 'val':
            relative_path = self.data_frame_val.iloc[idx, self.data_frame_val.columns.get_loc('relative_path')]
            # img_name = os.path.basename(relative_path)

            img_name = relative_path.replace('output/', '')
            img_path = os.path.join(self.root_dir, img_name)
            label = self.data_frame_val.iloc[idx, self.data_frame_val.columns.get_loc('label')]
            int_label = self.label_to_int[label]
            image = self.load_image(img_path)
            return image, int_label, img_name
        elif self.csv_file:
            relative_path = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('relative_path')]
            # img_name = os.path.basename(relative_path)

            img_name = relative_path.replace('output/', '')

            img_path = os.path.join(self.root_dir, img_name)
            label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('label')]
            int_label = self.label_to_int[label]
            image = self.load_image(img_path)
            return image, int_label, img_name
        else:
            filenames = [filename for filename in os.listdir(self.root_dir) if self.is_permitted_format(filename)]
            img_name = filenames[idx]
            img_path = os.path.join(self.root_dir, img_name)
            image = self.load_image(img_path)
            return image, '', img_name

    def load_image(self, img_path):
        image = Image.open(img_path)
        if image.mode not in ['L', 'RGB']:
            # Convert other modes like 'RGBA' or 'P' to 'RGB'
            image = image.convert('RGB')
        if self.gray:
            if image.mode != 'L':
                # Convert to grayscale
                image = image.convert('L')
        else:
            if image.mode != 'RGB':
                # Convert grayscale to RGB by repeating the grayscale channel three times
                image = image.convert('L')
                image = Image.merge("RGB", (image, image, image))

        if self.transform is not None:
            if self.phase == 'train' and self.transform.__len__() == 3:
                sample1 = self.transform[0](image)
                sample2 = self.transform[1](image)
                sample3 = self.transform[2](image)
                return [sample1, sample2, sample3]
            else:
                return self.transform(image)

        return image

    def is_permitted_format(self, filename):
        if self.permitted_formats is None:
            return True
        _, ext = os.path.splitext(filename)
        return ext.lower() in self.permitted_formats

    def get_string_label(self, int_label):
        if self.label_to_int is not None:
            return list(self.label_to_int.keys())[list(self.label_to_int.values()).index(int_label)]
        else:
            return None
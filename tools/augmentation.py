import numpy as np
from PIL import Image
from torchvision.transforms.functional import resized_crop
import torch
from torchvision import transforms


class RandomZoomIn:
    def __init__(self, zoom_range=(0.8, 1.0)):
        self.zoom_range = zoom_range

    def __call__(self, img):
        zoom_factor = torch.FloatTensor(1).uniform_(self.zoom_range[0], self.zoom_range[1]).item()

        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        left = (width - new_width) // 2
        top = (height - new_height) // 2

        img = resized_crop(img, top, left, new_height, new_width, (height, width))
        return img


class RandomZoomOut:
    def __init__(self, zoom_range=(1.0, 1.2)):
        self.zoom_range = zoom_range

    def __call__(self, img):
        zoom_factor = torch.FloatTensor(1).uniform_(self.zoom_range[0], self.zoom_range[1]).item()

        width, height = img.size
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        img = resized_crop(img, 0, 0, new_height, new_width, (height, width))
        return img


class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        img_array = np.array(img)
        noise = np.random.normal(0, self.std, img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        return Image.fromarray(noisy_img.astype(np.uint8))


class ResizeAndPad:
    def __init__(self, target_size):
        self.target_height, self.target_width = target_size

    def __call__(self, img):
        # Resize the image to maintain aspect ratio
        aspect_ratio = img.width / img.height
        if img.width < img.height:
            new_height = self.target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)

        img = transforms.Resize((new_height, new_width))(img)

        # Calculate padding needed
        pad_height = (self.target_height - new_height + 1) // 2 if new_height < self.target_height else 0
        pad_width = (self.target_width - new_width + 1) // 2 if new_width < self.target_width else 0

        # Pad the image to ensure it's exactly the target size
        img = transforms.Pad((pad_width, pad_height), fill=0, padding_mode='constant')(img)

        # If the image becomes larger due to rounding, we center crop
        img = transforms.CenterCrop((self.target_height, self.target_width))(img)

        return img
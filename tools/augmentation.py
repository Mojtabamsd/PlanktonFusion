import numpy as np
from PIL import Image


class RandomZoomIn:
    def __init__(self, zoom_range=(0.8, 1.0)):
        self.zoom_range = zoom_range

    def __call__(self, img):
        zoom_factor = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return img.crop((left, top, right, bottom))


class RandomZoomOut:
    def __init__(self, zoom_range=(1.0, 1.2)):
        self.zoom_range = zoom_range

    def __call__(self, img):
        zoom_factor = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        width, height = img.size
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        return img.resize((new_width, new_height), Image.BILINEAR)


class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        img_array = np.array(img)
        noise = np.random.normal(0, self.std, img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        return Image.fromarray(noisy_img.astype(np.uint8))
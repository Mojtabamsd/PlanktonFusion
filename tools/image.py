import os
from PIL import Image
import numpy as np


def save_img(img_batch, batch_idx, epoch, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    # Save every image in the current batch
    for i in range(img_batch.size(0)):
        image = img_batch[i].cpu().numpy().transpose((1, 2, 0))
        image = (image * 255).astype('uint8')
        image = np.squeeze(image)
        image_pil = Image.fromarray(image)

        filename = f"image_{epoch}_{batch_idx}_{i}.png"
        file_path = os.path.join(save_dir, filename)

        # Save the image
        image_pil.save(file_path)
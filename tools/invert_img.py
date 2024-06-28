import os
import argparse
import numpy as np
from PIL import Image
import time


def invert_images_in_directory(image_path):
    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)

        if os.path.isfile(file_path):
            temp_file_path = file_path + ".temp" + os.path.splitext(file_path)[1]
            try:
                with Image.open(file_path) as img:
                    img_gray = img.convert("L")
                    img_array = np.array(img_gray)
                    max_value = np.iinfo(img_array.dtype).max
                    inverted_array = max_value - img_array
                    inverted_img = Image.fromarray(inverted_array)
                    inverted_img.save(temp_file_path)
                    img.close()  # Explicitly close the file handle

                # time.sleep(0.1)  # Short delay to ensure file is released
                os.remove(file_path)
                os.rename(temp_file_path, file_path)
                # print(f"Inverted image saved: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)  # Cleanup if the process fails


def main():
    parser = argparse.ArgumentParser(description="Invert images in a directory.")
    parser.add_argument("image_path", type=str, help="Path to the directory containing images to be inverted.")
    args = parser.parse_args()

    invert_images_in_directory(args.image_path)

    # image_path = r'D:\mojmas\files\data\result_sampling\test\prediction20240308121300\detritus'
    # invert_images_in_directory(image_path)


if __name__ == "__main__":
    main()

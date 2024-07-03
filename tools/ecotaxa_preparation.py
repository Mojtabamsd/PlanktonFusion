import os
import argparse
import pandas as pd
import shutil
from zipfile import ZipFile
from PIL import Image
import numpy as np


def load_img(image_path, invert_img):
    """Load an image, convert to grayscale, invert it, and return the inverted image."""
    img = Image.open(image_path)
    img_gray = img.convert("L")
    if invert_img:
        img_array = np.array(img_gray)
        max_value = np.iinfo(img_array.dtype).max
        inverted_array = max_value - img_array
        inverted_img = Image.fromarray(inverted_array)
        return inverted_img
    else:
        return img_gray


def process_images_and_data(base_folder, output_folder, invert_img=True):
    """Process all images in subdirectories, save them as inverted PNGs in the output folder,
    and collect data for DataFrame."""

    img_file_names = ['[t]']
    object_ids = ['[t]']
    object_annotation_categories = ['[t]']

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name != 'output':
            for img_file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, img_file)
                if img_file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    inverted_img = load_img(file_path, invert_img)
                    base_filename = os.path.splitext(img_file)[0]
                    png_filename = base_filename + '.png'
                    png_path = os.path.join(output_folder, png_filename)
                    inverted_img.save(png_path, format='PNG')

                    img_file_names.append(png_filename)
                    object_ids.append(base_filename)
                    object_annotation_categories.append(folder_name)

    return img_file_names, object_ids, object_annotation_categories


def create_dataframe(data, sample_identifier):
    """Create Ecotaxa friendly DataFrame from the provided data dictionary."""

    df = pd.DataFrame(data)
    df['object_annotation_date'] = ['20240701'] * len(df)
    df['object_annotation_date'][0] = '[t]'

    df['object_annotation_time'] = ['00:00:00'] * len(df)
    df['object_annotation_time'][0] = '[t]'

    df['object_annotation_person_name'] = ['Mojtaba Masoudi'] * len(df)
    df['object_annotation_person_name'][0] = '[t]'

    df['object_annotation_person_email'] = ['masoudi.m1991@gmail.com'] * len(df)
    df['object_annotation_person_email'][0] = '[t]'

    df['object_annotation_status'] = ['predicted'] * len(df)
    df['object_annotation_status'][0] = '[t]'

    df['sample_id'] = ['st' + sample_identifier] * len(df)
    df['sample_id'][0] = '[t]'

    df['object_annotation_category'] = df['object_annotation_category'].replace('Copepoda', 'Copepoda<Maxillopoda')
    df['object_annotation_category'] = df['object_annotation_category'].replace('Cnidaria', 'Cnidaria<Metazoa')
    df['object_annotation_category'] = df['object_annotation_category'].replace('Ctenophora', 'Ctenophora<Metazoa')
    df['object_annotation_category'] = df['object_annotation_category'].replace('Crustacea_others', 'Crustacea')
    df['object_annotation_category'] = df['object_annotation_category'].replace('other_living', 'other<living')
    return df


def save_data_and_zip(output_folder, base_folder, data, sample_identifier):
    """Save the data as a TSV file and zip the output folder."""
    df = create_dataframe(data, sample_identifier)
    tsv_file_name = 'ecotaxa_' + os.path.basename(base_folder) + '.tsv'
    tsv_path = os.path.join(output_folder, tsv_file_name)
    df.to_csv(tsv_path, sep='\t', index=False)

    zip_name = os.path.join(base_folder, 'output.zip')
    with ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    return zip_name


def clean_directory(base_folder):
    """Remove all subdirectories in the base folder except 'output'."""
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name != 'output':
            shutil.rmtree(folder_path)


def dataframe_preparation(base_folder, sample_identifier):

    output_folder = os.path.join(base_folder, 'output')
    os.makedirs(output_folder, exist_ok=True)
    img_file_names, object_ids, object_annotation_categories = process_images_and_data(base_folder, output_folder)

    # Prepare data dictionary
    data = {
        'img_file_name': img_file_names,
        'object_id': object_ids,
        'object_annotation_category': object_annotation_categories,
    }

    # Save data and zip contents
    zip_file_path = save_data_and_zip(output_folder, base_folder, data, sample_identifier)
    clean_directory(base_folder)


def main():
    parser = argparse.ArgumentParser(description="Prepare classifier output for Ecotaxa website.")
    parser.add_argument("-i", "--image_path", type=str, help="Path to the directory containing images to be prepared.")
    parser.add_argument("-s", "--sample_id", type=str, default='01', help="Sample id")
    args = parser.parse_args()

    dataframe_preparation(args.image_path, args.sample_id)

    # image_path = r'D:\mojmas\files\data\result_sampling\test\prediction20240308121300 - Copy'
    # dataframe_preparation(image_path, sample_identifier=10)


if __name__ == "__main__":
    main()

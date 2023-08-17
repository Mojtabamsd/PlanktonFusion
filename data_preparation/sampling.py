import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import sys
from configs.config import Configuration
from tools.console import Console
from data_preparation.sampling_tools import (
    sampling_uniform,
    sampling_stratified,
    sampling_fixed_number,
    load_uvp5,
    load_uvp6,
    load_uvp6_from_csv,
    copy_image_from_df,
    report_csv
)


def sampling(config_path):

    Console.info("Sampling started at", datetime.datetime.now())
    config = Configuration(config_path)

    print("Data Path UVP5:", config.sampling.path_uvp5)
    print("Data Path UVP6:", config.sampling.path_uvp6)
# dir_uvp5, dir_uvp6, dir_output, which_uvp, class_type, sampling_method, target_size

    if config.sampling.uvp_type == 'UVP5':
        # load uvp5
        df1 = load_uvp5(config.sampling.path_uvp5)
        df2 = None
    elif config.sampling.uvp_type == 'UVP6':
        # load uvp6
        df2 = load_uvp6_from_csv(config.sampling.path_uvp6)
        df1 = None
    elif config.sampling.uvp_type == 'BOTH':
        df1 = load_uvp5(config.sampling.path_uvp5)
        df2 = load_uvp6_from_csv(config.sampling.path_uvp6)
    else:
        print("Please select correct parameter for which_uvp")
        sys.exit()

    # regrouping
    ren = pd.read_csv("./data_preparation/regrouping.csv")
    if config.sampling.num_class == 13:
        merge_dict = dict(zip(ren['taxon'], ren['regrouped2']))
    elif config.sampling.num_class == 25:
        merge_dict = dict(zip(ren['taxon'], ren['regrouped1']))
    else:
        print("Please select correct parameter for class_type")
        sys.exit()

    if df1 is not None:
        df1['label'] = df1['label'].map(merge_dict).fillna(df1['label'])
        if config.sampling.sampling_method == 'fixed':
            df1_sample = sampling_fixed_number(df1, config.sampling.sampling_percent_uvp5)
        elif config.sampling.sampling_method == 'uniform':
            df1_sample = sampling_uniform(df1, config.sampling.sampling_percent_uvp5)
        elif config.sampling.sampling_method == 'stratified':
            df1_sample = sampling_stratified(df1, config.sampling.sampling_percent_uvp5)
        else:
            print("Please select correct parameter for sampling_method")
            sys.exit()
    else:
        df1_sample = None

    if df2 is not None:
        df2['label'] = df2['label'].map(merge_dict).fillna(df2['label'])
        if config.sampling.sampling_method == 'fixed':
            df2_sample = sampling_fixed_number(df2, config.sampling.sampling_percent_uvp6)
        elif config.sampling.sampling_method == 'uniform':
            df2_sample = sampling_uniform(df2, config.sampling.sampling_percent_uvp6)
        elif config.sampling.sampling_method == 'stratified':
            df2_sample = sampling_stratified(df2, config.sampling.sampling_percent_uvp6)
        else:
            print("Please select correct parameter for sampling_method")
            sys.exit()
    else:
        df2_sample = None

    # create sampling output
    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    output_folder = Path(config.sampling.path_output)
    sampling_path = output_folder / ("sampling" + time_str)
    # if not sampling_path.exists():
    sampling_path.mkdir(parents=True, exist_ok=True)
    sampling_path_images = sampling_path / ("output")
    sampling_path_images.mkdir(parents=True, exist_ok=True)

    # Loop through the image paths and copy the images to the target directory
    if df1_sample is not None:
        copy_image_from_df(df1_sample, sampling_path_images, config.sampling.target_size, cutting_ruler=True, invert_img=True)
    if df2_sample is not None:
        copy_image_from_df(df2_sample, sampling_path_images, config.sampling.target_size, cutting_ruler=False, invert_img=False)

    # merge two dataframe
    df = pd.concat([df1_sample, df2_sample])

    # shuffle and remove redundant columns
    df = df.drop('path', axis=1)
    df = df.sample(frac=1, random_state=np.random.seed())
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.reset_index()
    df = df.replace('NaN', 0)

    selected_columns = df[['index', 'profile_id', 'object_id', 'depth', 'lat', 'lon', 'datetime', 'uvp_model',
                           'label', 'relative_path']]

    csv_path = sampling_path / ("sampled_images.csv")
    selected_columns.to_csv(csv_path)

    report_csv(df1, df1_sample, df2, df2_sample, sampling_path)


# if __name__ == "__main__":
#     dir_uvp5 = r'D:\mojmas\files\data\UVP5_images_dataset'
#     # dir_uvp6 = r'D:\mojmas\files\data\UVP6Net'
#     dir_uvp6 = r'D:\mojmas\files\data\csv_history\sampled_images5.csv'
#     dir_output = r'D:\mojmas\files\data\result_sampling'
#
#     which_uvp = 1  # '0' uvp5, '1' uvp6, '2' both uvp merge
#     class_type = 0  # '0' means 13 class or '1' means 25 classes
#
#     # '0' sample fixed number, '1' sample uniform percent from each class, '2' sample stratified from each uvps
#     sampling_method = 0
#
#     # if sampling fixed numbers
#     sampling_percent_uvp5 = 100
#     sampling_percent_uvp6 = 100
#
#     # # if sampling stratified
#     # sampling_percent_uvp5 = 0.0051  # all=9,884,798
#     # sampling_percent_uvp6 = 0.08  # all=634,459
#
#     target_size = [227, 227]
#
#     sampling(dir_uvp5, dir_uvp6, dir_output, which_uvp, class_type, sampling_method, target_size)
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import sys
from configs.config import Configuration
from tools.console import Console
from data_preparation.sampling_tools import (
    sampling_uniform,
    sampling_uniform_test,
    sampling_stratified,
    sampling_stratified_test,
    sampling_fixed_number,
    sampling_fixed_number_test,
    load_uvp5,
    load_uvp6,
    load_uvp6_from_csv,
    copy_image_from_df,
    report_csv
)


def sampling(config_path):

    config = Configuration(config_path)
    output_folder = Path(config.sampling.path_output)
    console = Console(output_folder)
    console.info("Sampling started at", datetime.datetime.now())

    console.info("Data Path UVP5:", config.sampling.path_uvp5)
    console.info("Data Path UVP6:", config.sampling.path_uvp6)

    if config.sampling.uvp_type == 'UVP5':
        # load uvp5
        df1 = load_uvp5(config.sampling.path_uvp5)
        df1['label'] = df1['label'].str.replace('<', '_')
        df2 = None
    elif config.sampling.uvp_type == 'UVP6':
        # load uvp6
        if config.sampling.path_uvp6_csv:
            df2 = load_uvp6_from_csv(config.sampling.path_uvp6_csv)
        else:
            df2 = load_uvp6(config.sampling.path_uvp6)
        df2['label'] = df2['label'].str.replace('<', '_')
        df1 = None
    elif config.sampling.uvp_type == 'BOTH':
        df1 = load_uvp5(config.sampling.path_uvp5)
        df1['label'] = df1['label'].str.replace('<', '_')

        if config.sampling.path_uvp6_csv:
            df2 = load_uvp6_from_csv(config.sampling.path_uvp6_csv)
        else:
            df2 = load_uvp6(config.sampling.path_uvp6)
        df2['label'] = df2['label'].str.replace('<', '_')
    else:
        console.error("Please select correct parameter for which_uvp")
        sys.exit()

    # regrouping
    ren = pd.read_csv("./data_preparation/regrouping.csv")
    if config.sampling.num_class == 13:
        merge_dict = dict(zip(ren['taxon'], ren['regrouped2']))
    elif config.sampling.num_class == 25:
        merge_dict = dict(zip(ren['taxon'], ren['regrouped1']))
    elif config.sampling.num_class == 2:
        merge_dict = dict(zip(ren['taxon'], ren['regrouped3']))
    else:
        console.error("Please select correct parameter for class_type")
        sys.exit()

    if df1 is not None:
        df1['label'] = df1['label'].map(merge_dict).fillna(df1['label'])
        if config.sampling.create_folder:
            df1['relative_path'] = df1.apply(lambda row: f"output/{row['label']}/"
                                             f"{row['relative_path'].split('/')[1]}", axis=1)

        if config.sampling.test_dataset_sampling == 'fixed':
            df1_sample_test, df1_train = sampling_fixed_number_test(df1, config.sampling.test_percent_uvp5)
        elif config.sampling.test_dataset_sampling == 'uniform':
            df1_sample_test, df1_train = sampling_uniform_test(df1, config.sampling.test_percent_uvp5)
        elif config.sampling.test_dataset_sampling == 'stratified':
            df1_sample_test, df1_train = sampling_stratified_test(df1, config.sampling.test_percent_uvp5)
        else:
            console.error("Please select correct parameter for test_dataset_sampling")
            sys.exit()

        if config.sampling.sampling_method == 'fixed':
            df1_sample = sampling_fixed_number(df1_train, config.sampling.sampling_percent_uvp5)
        elif config.sampling.sampling_method == 'uniform':
            df1_sample = sampling_uniform(df1_train, config.sampling.sampling_percent_uvp5)
        elif config.sampling.sampling_method == 'stratified':
            df1_sample = sampling_stratified(df1_train, config.sampling.sampling_percent_uvp5)
        else:
            console.error("Please select correct parameter for sampling_method")
            sys.exit()
    else:
        df1_sample = None
        df1_sample_test = None
        df1_train = df1

    if df2 is not None:
        df2['label'] = df2['label'].map(merge_dict).fillna(df2['label'])
        if config.sampling.create_folder:
            df2['relative_path'] = df2.apply(lambda row: f"output/{row['label']}/"
                                             f"{row['relative_path'].split('/')[1]}", axis=1)

        if config.sampling.test_dataset_sampling == 'fixed':
            df2_sample_test, df2_train = sampling_fixed_number_test(df2, config.sampling.test_percent_uvp6)
        elif config.sampling.test_dataset_sampling == 'uniform':
            df2_sample_test, df2_train = sampling_uniform_test(df2, config.sampling.test_percent_uvp6)
        elif config.sampling.test_dataset_sampling == 'stratified':
            df2_sample_test, df2_train = sampling_stratified_test(df2, config.sampling.test_percent_uvp6)
        else:
            console.error("Please select correct parameter for test_dataset_sampling")
            sys.exit()

        if config.sampling.sampling_method == 'fixed':
            df2_sample = sampling_fixed_number(df2_train, config.sampling.sampling_percent_uvp6)
        elif config.sampling.sampling_method == 'uniform':
            df2_sample = sampling_uniform(df2_train, config.sampling.sampling_percent_uvp6)
        elif config.sampling.sampling_method == 'stratified':
            df2_sample = sampling_stratified(df2_train, config.sampling.sampling_percent_uvp6)
        else:
            console.error("Please select correct parameter for sampling_method")
            sys.exit()
    else:
        df2_sample = None
        df2_sample_test = None
        df2_train = df2

    # create sampling output
    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    sampling_path_train = output_folder / ("sampling" + time_str) / "train"
    # if not sampling_path_train.exists():
    sampling_path_train.mkdir(parents=True, exist_ok=True)
    sampling_path_images_train = sampling_path_train / ("output")
    sampling_path_images_train.mkdir(parents=True, exist_ok=True)

    sampling_path_test = output_folder / ("sampling" + time_str) / "test"
    sampling_path_test.mkdir(parents=True, exist_ok=True)
    sampling_path_images_test = sampling_path_test / ("output")
    sampling_path_images_test.mkdir(parents=True, exist_ok=True)

    # Loop through the image paths and copy the images to the target directory
    if df1_sample is not None:
        copy_image_from_df(df1_sample, sampling_path_images_train, config.sampling.target_size,
                           cutting_ruler=True, invert_img=True)

        copy_image_from_df(df1_sample_test, sampling_path_images_test, config.sampling.target_size,
                           cutting_ruler=True, invert_img=True)

    if df2_sample is not None:
        copy_image_from_df(df2_sample, sampling_path_images_train, config.sampling.target_size,
                           cutting_ruler=False, invert_img=True)

        copy_image_from_df(df2_sample_test, sampling_path_images_test, config.sampling.target_size,
                           cutting_ruler=False, invert_img=True)

    # merge two dataframe for train
    df_train = pd.concat([df1_sample, df2_sample])

    # shuffle and remove redundant columns
    df_train = df_train.drop('path', axis=1)
    df_train = df_train.sample(frac=1, random_state=np.random.seed())
    df_train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
    df_train = df_train.reset_index()
    df_train = df_train.replace('NaN', 0)

    selected_columns = df_train[['index', 'profile_id', 'object_id', 'depth', 'lat', 'lon', 'datetime', 'uvp_model',
                                'label', 'relative_path']]

    csv_path = sampling_path_train / ("sampled_images.csv")
    selected_columns.to_csv(csv_path)

    report_csv(df1_train, df1_sample, df2_train, df2_sample, sampling_path_train)

    # merge two dataframe for test
    df_test = pd.concat([df1_sample_test, df2_sample_test])

    # shuffle and remove redundant columns
    df_test = df_test.drop('path', axis=1)
    df_test = df_test.sample(frac=1, random_state=np.random.seed())
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    df_test = df_test.reset_index()
    df_test = df_test.replace('NaN', 0)

    selected_columns = df_test[['index', 'profile_id', 'object_id', 'depth', 'lat', 'lon', 'datetime', 'uvp_model',
                           'label', 'relative_path']]

    csv_path = sampling_path_test / ("sampled_images.csv")
    selected_columns.to_csv(csv_path)

    report_csv(df1, df1_sample_test, df2, df2_sample_test, sampling_path_test)
    # console.save_log(sampling_path_train)


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
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
    report_csv,
    create_dataframe_from_folder
)


def sampling_syn(config_path):

    config = Configuration(config_path)
    output_folder = Path(config.sampling_syn.path_output)
    console = Console(output_folder)
    console.info("sampling synthetic started at", datetime.datetime.now())

    console.info("Data Path UVP6:", config.sampling_syn.path_uvp6)

    df1 = create_dataframe_from_folder(config.sampling_syn.path_syn)
    df1_sample = df1
    df1_train = df1
    df1_sample_test = None

    if config.sampling_syn.uvp_type == 'UVP6':
        # load uvp6
        if config.sampling_syn.path_uvp6_csv:
            df2 = load_uvp6_from_csv(config.sampling_syn.path_uvp6_csv)
        else:
            df2 = load_uvp6(config.sampling_syn.path_uvp6)
        df2['label'] = df2['label'].str.replace('<', '_')
    else:
        console.error("Please select correct parameter for which_uvp")
        sys.exit()

    labels_included = ['Amphipoda', 'Aulacanthidae', 'Calanidae']
    if df2 is not None:
        df2 = df2[df2['label'].isin(labels_included)]
        if config.sampling_syn.create_folder:
            df2['relative_path'] = df2.apply(lambda row: f"output/{row['label']}/"
                                             f"{row['relative_path'].split('/')[1]}", axis=1)

        # # to exclude of taxon
        # df2 = df2[df2['label'] != 'detritus']

        if config.sampling_syn.test_dataset_sampling == 'fixed':
            df2_sample_test, df2_train = sampling_fixed_number_test(df2, config.sampling_syn.test_percent_uvp6)
        elif config.sampling_syn.test_dataset_sampling == 'uniform':
            df2_sample_test, df2_train = sampling_uniform_test(df2, config.sampling_syn.test_percent_uvp6)
        elif config.sampling_syn.test_dataset_sampling == 'stratified':
            df2_sample_test, df2_train = sampling_stratified_test(df2, config.sampling_syn.test_percent_uvp6)
        else:
            console.error("Please select correct parameter for test_dataset_sampling")
            sys.exit()

        if config.sampling_syn.sampling_method == 'fixed':
            df2_sample = sampling_fixed_number(df2_train, config.sampling_syn.sampling_percent_uvp6)
        elif config.sampling_syn.sampling_method == 'uniform':
            df2_sample = sampling_uniform(df2_train, config.sampling_syn.sampling_percent_uvp6)
        elif config.sampling_syn.sampling_method == 'stratified':
            df2_sample = sampling_stratified(df2_train, config.sampling_syn.sampling_percent_uvp6)
        else:
            console.error("Please select correct parameter for sampling_method")
            sys.exit()
    else:
        df2_sample = None
        df2_sample_test = None
        df2_train = df2

    # create sampling_syn output
    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    sampling_path_train = output_folder / ("sampling_syn" + time_str) / "train"
    # if not sampling_path_train.exists():
    sampling_path_train.mkdir(parents=True, exist_ok=True)
    sampling_path_images_train = sampling_path_train / "output"
    sampling_path_images_train.mkdir(parents=True, exist_ok=True)

    sampling_path_test = output_folder / ("sampling_syn" + time_str) / "test"
    sampling_path_test.mkdir(parents=True, exist_ok=True)
    sampling_path_images_test = sampling_path_test / "output"
    sampling_path_images_test.mkdir(parents=True, exist_ok=True)

    # Loop through the image paths and copy the images to the target directory
    if df1_sample is not None:
        copy_image_from_df(df1_sample, sampling_path_images_train, config.sampling_syn.target_size,
                           cutting_ruler=True, invert_img=True)

    if df2_sample is not None:
        copy_image_from_df(df2_sample, sampling_path_images_train, config.sampling_syn.target_size,
                           cutting_ruler=False, invert_img=True)

        copy_image_from_df(df2_sample_test, sampling_path_images_test, config.sampling_syn.target_size,
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

    report_csv(df1_train, df1_sample, df2_train, df2_sample, sampling_path_train, syn=True)

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

    report_csv(df1_sample_test, df1_sample_test, df2, df2_sample_test, sampling_path_test, syn=True)
    # console.save_log(sampling_path_train)
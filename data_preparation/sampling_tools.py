import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


# def sampling_stratified(df, percent):
#     # shffle data
#     df = df.sample(frac=1, random_state=np.random.seed())
#     grouped = df.groupby('label')
#
#     # Initialize an empty DataFrame to store the sampled rows
#     sampled = pd.DataFrame()
#
#     # For each group type, select 10 percent of the rows randomly
#     for group_name, group_data in grouped:
#         sample = group_data.sample(frac=percent, random_state=42)
#         sampled = pd.concat([sampled, sample])
#
#     return sampled


def sampling_stratified(df, percent):
    # Shuffle data
    df = df.sample(frac=1, random_state=np.random.seed())

    # Split data into features (X) and labels (y)
    y = df['label']

    # Initialize an empty DataFrame to store the sampled rows
    sampled = pd.DataFrame()

    # Create a StratifiedShuffleSplit object to sample 10 percent of the data
    sss = StratifiedShuffleSplit(n_splits=1, train_size=percent, random_state=42)

    # Iterate over the indices for the training set (10 percent)
    for train_index, _ in sss.split(df, y):
        sample = df.iloc[train_index]
        sampled = pd.concat([sampled, sample])

    return sampled


def sampling_stratified_test(df, percent):
    # Shuffle data
    df = df.sample(frac=1, random_state=np.random.seed())

    # Split data into labels (y)
    y = df['label']

    # Initialize an empty DataFrame to store the sampled rows
    sampled = pd.DataFrame()

    # Create a StratifiedShuffleSplit object to sample 10 percent of the data
    sss = StratifiedShuffleSplit(n_splits=1, train_size=percent, random_state=42)

    # Iterate over the indices for the training set (10 percent)
    for train_index, _ in sss.split(df, y):
        sample = df.iloc[train_index]
        sampled = pd.concat([sampled, sample])

    # Create a DataFrame containing the original data without the sampled rows
    original_without_sampled = df.drop(sampled.index)

    return sampled, original_without_sampled


def sampling_uniform(df, percent):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())

    group_sizes = df.groupby('label').size()
    group_sample_sizes = (group_sizes / group_sizes.sum() * len(df) * percent).round().astype(int)
    min_sample_size = group_sample_sizes.min()

    # Sample the same number of data points from each class
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min_sample_size, replace=False) if len(x) < min_sample_size else x.sample(n=min_sample_size,
                                                                                                      replace=False))

    return sampled_df


def sampling_uniform_test(df, percent):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())

    group_sizes = df.groupby('label').size()
    group_sample_sizes = (group_sizes / group_sizes.sum() * len(df) * percent).round().astype(int)
    min_sample_size = group_sample_sizes.min()

    # Sample the same number of data points from each class
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min_sample_size, replace=False) if len(x) < min_sample_size else x.sample(n=min_sample_size,
                                                                                                      replace=False))

    # Create a DataFrame containing the original data without the sampled rows
    original_without_sampled = df.drop(sampled_df.index)

    return sampled_df, original_without_sampled


def sampling_fixed_number(df, fixed_number):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=fixed_number, replace=False) if len(x) > fixed_number else x.sample(n=len(x), replace=False))

    return sampled_df


def sampling_fixed_number_test(df, fixed_number):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=fixed_number, replace=False) if len(x) > fixed_number else x.sample(n=len(x), replace=False))

    # Create a DataFrame containing the original data without the sampled rows
    original_without_sampled = df.drop(sampled_df.index)

    return sampled_df, original_without_sampled


def load_uvp5(path):
    o = pd.read_csv((path + r'\final\objects.tsv.gz'), sep='\t',
                    usecols=['profile_id', 'object_id', 'group', 'depth', ])
    s = pd.read_csv((path + r'\final\samples.tsv.gz'), sep='\t', index_col='profile_id',
                    usecols=['profile_id', 'lat',
                             'lon', 'datetime',
                             'pixel_size',
                             'uvp_model'])

    df = o.join(s, on='profile_id')
    df = df.rename(columns={'group': 'label', })

    # dicard ZD values
    df = df[df['uvp_model'] != 'ZD']

    # create path column uvp5
    df['profile_id'] = df['profile_id'].astype(int)
    df['object_id'] = df['object_id'].astype(int)
    df['path'] = path + '\\images\\' + df['profile_id'].astype(str) + '\\' + df['object_id'].astype(str) + '.jpg'
    df['relative_path'] = 'output\\' + df['object_id'].astype(str) + '.jpg'
    df['relative_path'] = df['relative_path'].str.replace(r'\s*\\+\s*', '/', regex=True)

    return df


def load_uvp6(path):
    df = pd.read_csv((path + r'\taxa.csv.gz'), sep=',', usecols=['objid', 'taxon'])
    df['label'] = df['taxon']
    df = df.rename(columns={'objid': 'object_id', })

    df['uvp_model'] = ['UVP6'] * len(df)
    df['profile_id'] = ['NaN'] * len(df)
    df['depth'] = ['NaN'] * len(df)
    df['lat'] = ['NaN'] * len(df)
    df['lon'] = ['NaN'] * len(df)
    df['datetime'] = ['NaN'] * len(df)
    df['pixel_size'] = ['NaN'] * len(df)

    df['taxon'] = df['taxon'].str.replace('<', '_')
    df['taxon'] = df['taxon'].str.replace(' ', '_')

    df['object_id'] = df['object_id'].astype(int)
    df['path'] = path + '\\imgs\\' + df['taxon'].astype(str) + '\\' + df['object_id'].astype(str) + '.png'
    df['relative_path'] = 'output/' + df['object_id'].astype(str) + '.png'

    df = df.drop('taxon', axis=1)

    return df


def load_uvp6_from_csv(csv_path):

    df = pd.read_csv(csv_path, sep=',').reset_index(drop=True, inplace=False)
    df = df.rename(columns={'groundthruth': 'label', })

    df['label'] = df['label'].str.replace('fi', 'fiber')
    df['label'] = df['label'].str.replace('_', '<')
    df['label'] = df['label'].str.replace('copepoda<eggs', 'copepoda eggs')
    df['label'] = df['label'].str.replace('Hydrozoa', 'Hydrozoa_others')
    df['label'] = df['label'].str.replace('Mollusca', 'Mollusca_others')
    df['label'] = df['label'].str.replace('fiberber', 'detritus')
    df['label'] = df['label'].str.replace('fiberlament', 'detritus')
    df['label'] = df['label'].str.replace('Creseis acicula', 'Creseis')
    df['label'] = df['label'].str.replace('antenna<Calanoida', 'Calanoida')
    df = df[df['label'] != 'Botrynema']

    df = df.rename(columns={'depth [m]': 'depth', })
    df = df.rename(columns={'latitude [deg]': 'lat', })
    df = df.rename(columns={'longitude [deg]': 'lon', })
    df = df.rename(columns={'timestamp [s]': 'datetime', })
    df['profile_id'] = 0
    df['object_id'] = 0

    # create path column uvp6
    df['uvp_model'] = ['UVP6'] * len(df)
    dir1_ = r'D:\mojmas\files\data'
    df['relative_path'] = df['relative_path'].str.replace(r'\\', '/')
    df['path'] = dir1_ + r'\\' + df['relative_path'].astype(str)
    return df


def copy_image_from_df(df, out_dir, target_size=None, cutting_ruler=False, invert_img=True):

    rows_to_drop = []
    for index, row in tqdm(df.iterrows()):
        image_path = row['path']
        if not os.path.exists(image_path):
            rows_to_drop.append(index)  # Store index of row to drop
            continue
        # image_filename = os.path.basename(image_path)
        image_filename = row['relative_path']
        image_filename = image_filename.replace('output/', '', 1)
        target_path = os.path.join(out_dir, image_filename)
        dir_path = os.path.dirname(target_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        img = Image.open(image_path)

        if cutting_ruler:
            # crop 31px from bottom
            width, height = img.size
            right = width
            bottom = height - 31
            cropped_img = img.crop((0, 0, right, bottom))
        else:
            cropped_img = img

        if invert_img:
            # invert image
            img_gray = cropped_img.convert("L")
            img_array = np.array(img_gray)
            max_value = np.iinfo(img_array.dtype).max
            inverted_array = max_value - img_array
            inverted_img = Image.fromarray(inverted_array)
        else:
            inverted_img = cropped_img
        # resize image
        if target_size is not None and target_size != 'None':
            resized_image = inverted_img.resize((target_size[0], target_size[1]), resample=Image.Resampling.LANCZOS)
        else:
            resized_image = inverted_img
        resized_image.save(target_path)

    df.drop(rows_to_drop, inplace=True)
    return df


def report_csv(df1, df1_sample, df2, df2_sample, sampling_path=None, syn=False):
    # report
    if df1 is None:
        res2 = df2.groupby(['label']).size()
        res2_sample = df2_sample.groupby(['label']).size()

        res1 = res2.copy()
        res1[:] = 0
        res1_sample = res2_sample.copy()
        res1_sample[:] = 0

    elif df2 is None:
        res1 = df1.groupby(['label']).size()
        res1_sample = df1_sample.groupby(['label']).size()

        res2 = res1.copy()
        res2[:] = 0
        res2_sample = res1_sample.copy()
        res2_sample[:] = 0

    else:
        res1 = df1.groupby(['label']).size()
        res1_sample = df1_sample.groupby(['label']).size()

        res2 = df2.groupby(['label']).size()
        res2_sample = df2_sample.groupby(['label']).size()

    if syn:
        series1 = pd.Series(res1, name='syn')
        series2 = pd.Series(res1_sample, name='syn_sample')
    else:
        series1 = pd.Series(res1, name='uvp5')
        series2 = pd.Series(res1_sample, name='uvp5_sample')
    series3 = pd.Series(res2, name='uvp6_sample')
    series4 = pd.Series(res2_sample, name='uvp6_sample')

    # Merge the two Series based on their index
    merged_series = pd.merge(series1, series2, left_index=True, right_index=True)
    merged_series = pd.merge(merged_series, series3, left_index=True, right_index=True, how='outer')
    merged_series = pd.merge(merged_series, series4, left_index=True, right_index=True, how='outer')

    csv_report_path = sampling_path / ("report.csv")
    merged_series.to_csv(csv_report_path)


def create_dataframe_from_folder(folder_path):
    data = {'label': [], 'path': [], 'relative_path': []}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            label = os.path.basename(root)
            file_path = os.path.join(root, file)

            relative_path = os.path.relpath(file_path, folder_path)
            relative_path = 'output/' + os.path.basename(relative_path)

            # Append data to the dictionary
            data['label'].append(label)
            data['path'].append(file_path)
            data['relative_path'].append(relative_path)

    df = pd.DataFrame(data)
    df['uvp_model'] = ['SYN'] * len(df)
    df['object_id'] = ['NaN'] * len(df)
    df['profile_id'] = ['NaN'] * len(df)
    df['depth'] = ['NaN'] * len(df)
    df['lat'] = ['NaN'] * len(df)
    df['lon'] = ['NaN'] * len(df)
    df['datetime'] = ['NaN'] * len(df)
    df['pixel_size'] = ['NaN'] * len(df)
    return df
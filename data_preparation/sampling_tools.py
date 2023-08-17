import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image


def sampling_stratified(df, percent):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())
    grouped = df.groupby('groundtruth')

    # Initialize an empty DataFrame to store the sampled rows
    sampled = pd.DataFrame()

    # For each group type, select 10 percent of the rows randomly
    for group_name, group_data in grouped:
        sample = group_data.sample(frac=percent, random_state=42)
        sampled = pd.concat([sampled, sample])

    return sampled


def sampling_uniform(df, percent):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())

    group_sizes = df.groupby('groundtruth').size()
    group_sample_sizes = (group_sizes / group_sizes.sum() * len(df) * percent).round().astype(int)
    min_sample_size = group_sample_sizes.min()

    # Sample the same number of data points from each class
    sampled_df = df.groupby('groundtruth', group_keys=False).apply(
        lambda x: x.sample(n=min_sample_size, replace=True) if len(x) < min_sample_size else x.sample(n=min_sample_size,
                                                                                                      replace=False))

    return sampled_df


def sampling_fixed_number(df, fixed_number):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())
    sampled_df = df.groupby('groundtruth', group_keys=False).apply(
        lambda x: x.sample(n=fixed_number, replace=True) if len(x) > fixed_number else x.sample(n=len(x), replace=False))

    return sampled_df


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
    df['relative_path'] = df['relative_path'].str.replace(r'\\', '/')

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
    df['relative_path'] = 'output\\' + df['object_id'].astype(str) + '.png'

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

    if target_size is None:
        target_size = [227, 227]

    for index, row in tqdm(df.iterrows()):
        image_path = row['path']
        image_filename = os.path.basename(image_path)
        target_path = os.path.join(out_dir, image_filename)
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
        resized_image = inverted_img.resize((target_size[0], target_size[1]), resample=Image.Resampling.LANCZOS)
        resized_image.save(target_path)


def report_csv(df1, df1_sample, df2, df2_sample, sampling_path=None):
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

    series1 = pd.Series(res1, name='uvp5')
    series2 = pd.Series(res1_sample, name='uvp5_sample')
    series3 = pd.Series(res2, name='uvp6_sample')
    series4 = pd.Series(res2_sample, name='uvp6_sample')

    # Merge the two Series based on their index
    merged_series = pd.merge(series1, series2, left_index=True, right_index=True)
    merged_series = pd.merge(merged_series, series3, left_index=True, right_index=True)
    merged_series = pd.merge(merged_series, series4, left_index=True, right_index=True)

    csv_report_path = sampling_path / ("report.csv")
    merged_series.to_csv(csv_report_path)
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
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
    df = df.rename(columns={'group': 'groundtruth', })

    # dicard ZD values
    df = df[df['uvp_model'] != 'ZD']

    return df


def load_uvp6(path):
    df = pd.read_csv((path + r'\taxa.csv.gz'), sep=',', usecols=['objid', 'taxon'])
    df['groundtruth'] = df['taxon']
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
    df['path'] = dir_uvp6 + '\\imgs\\' + df['taxon'].astype(str) + '\\' + df['object_id'].astype(str) + '.png'
    df['relative_path'] = 'output\\' + df['object_id'].astype(str) + '.png'

    df = df.drop('taxon', axis=1)

    return df


def load_uvp6_from_csv(csv_path):

    df = pd.read_csv(csv_path, sep=',').reset_index(drop=True, inplace=False)
    df = df.rename(columns={'groundthruth': 'groundtruth', })

    df['groundtruth'] = df['groundtruth'].str.replace('fi', 'fiber')
    df['groundtruth'] = df['groundtruth'].str.replace('_', '<')
    df['groundtruth'] = df['groundtruth'].str.replace('copepoda<eggs', 'copepoda eggs')
    df['groundtruth'] = df['groundtruth'].str.replace('Hydrozoa', 'Hydrozoa_others')
    df['groundtruth'] = df['groundtruth'].str.replace('Mollusca', 'Mollusca_others')
    df['groundtruth'] = df['groundtruth'].str.replace('fiberber', 'detritus')
    df['groundtruth'] = df['groundtruth'].str.replace('fiberlament', 'detritus')
    df['groundtruth'] = df['groundtruth'].str.replace('Creseis acicula', 'Creseis')
    df['groundtruth'] = df['groundtruth'].str.replace('antenna<Calanoida', 'Calanoida')
    df = df[df['groundtruth'] != 'Botrynema']

    df['uvp_model'] = ['UVP6'] * len(df)
    dir1_ = r'D:\mojmas\files\data'
    df['relative_path'] = df['relative_path'].str.replace(r'\\', '/')
    df['path'] = dir1_ + r'\\' + df['relative_path'].astype(str)

    return df



dir_uvp5 = r'D:\mojmas\files\data\UVP5_images_dataset'
# dir_uvp6 = r'D:\mojmas\files\data\UVP6Net'
dir_uvp6 = r'D:\mojmas\files\data\csv_history\sampled_images5.csv'
dir_output = r'D:\mojmas\files\data\result_sampling'


which_uvp = 0  # '0' uvp5, '1' uvp6, '2' both uvp merge
class_type = 0  # '0' means 13 class or '1' means 25 classes

stratified_flag = False
sampling_percent_uvp5 = 0.0051  # all=9,884,798
sampling_percent_uvp6 = 0.08  # all=634,459

uvp5_cut_flag = True

sample_fixed = True
if sample_fixed:
    sampling_percent_uvp5 = 100
    sampling_percent_uvp6 = 100

target_size = [227, 227]

# load uvp5
df1 = load_uvp5(dir_uvp5)
# load uvp6
df2 = load_uvp6_from_csv(dir_uvp6)


# regrouping
ren = pd.read_csv("../data_preparation/regrouping.csv")
merge_dict = dict(zip(ren['taxon'], ren['regrouped2']))

df1['groundtruth'] = df1['groundtruth'].map(merge_dict).fillna(df1['groundtruth'])
df2['groundtruth'] = df2['groundtruth'].map(merge_dict).fillna(df2['groundtruth'])

if sample_fixed:
    df1_sample = sampling_fixed_number(df1, sampling_percent_uvp5)
    df2_sample = sampling_fixed_number(df2, sampling_percent_uvp6)
elif stratified_flag:
    df1_sample = sampling_stratified(df1, sampling_percent_uvp5)
    df2_sample = sampling_stratified(df2, sampling_percent_uvp6)
else:
    df1_sample = sampling_uniform(df1, sampling_percent_uvp5)
    df2_sample = sampling_uniform(df2, sampling_percent_uvp6)


# create path column uvp5
df1_sample['profile_id'] = df1_sample['profile_id'].astype(int)
df1_sample['object_id'] = df1_sample['object_id'].astype(int)
df1_sample['path'] = dir_uvp5 + '\\images\\' + df1_sample['profile_id'].astype(str) + '\\' + df1_sample['object_id'].astype(str) + '.jpg'
df1_sample['relative_path'] = 'output\\' + df1_sample['object_id'].astype(str) + '.jpg'


# create path column uvp6


# create sampling output
time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
output_folder = Path(dir_output)
sampling_path = output_folder / ("sampling" + time_str)
# if not sampling_path.exists():
sampling_path.mkdir(parents=True, exist_ok=True)
sampling_path_images = sampling_path / ("output")
sampling_path_images.mkdir(parents=True, exist_ok=True)

# Loop through the image paths and copy the images to the target directory

# # uvp5
if uvp5_cut_flag:
    for index, row in tqdm(df1_sample.iterrows()):
        image_path = row['path']
        image_filename = os.path.basename(image_path)
        target_path = os.path.join(sampling_path_images, image_filename)
        # crop 31px from bottom
        img = Image.open(image_path)
        # crop image
        width, height = img.size
        right = width
        bottom = height - 31
        cropped_img = img.crop((0, 0, right, bottom))
        # invert image
        img_gray = cropped_img.convert("L")
        img_array = np.array(img_gray)
        max_value = np.iinfo(img_array.dtype).max
        inverted_array = max_value - img_array
        inverted_img = Image.fromarray(inverted_array)
        # resize image
        resized_image = inverted_img.resize((target_size[0], target_size[1]), resample=Image.Resampling.LANCZOS)
        resized_image.save(target_path)
else:
    for index, row in tqdm(df1_sample.iterrows()):
        image_path = row['path']
        image_filename = os.path.basename(image_path)
        target_path = os.path.join(sampling_path_images, image_filename)
        shutil.copy(image_path, target_path)

# uvp6
for index, row in tqdm(df2_sample.iterrows()):
    image_path = row['path']
    image_filename = os.path.basename(image_path)
    target_path = os.path.join(sampling_path_images, image_filename)
    img = Image.open(image_path)
    # invert image
    img_gray = img.convert("L")
    img_array = np.array(img_gray)
    max_value = np.iinfo(img_array.dtype).max
    inverted_array = max_value - img_array
    inverted_img = Image.fromarray(inverted_array)
    # resize image
    resized_image = inverted_img.resize((target_size[0], target_size[1]), resample=Image.Resampling.LANCZOS)
    resized_image.save(target_path)


# merge two dataframe
df = pd.concat([df1_sample, df2_sample])


# shtulle and remove redundant columns
df = df.drop('path', axis=1)
df = df.sample(frac=1, random_state=np.random.seed())
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.reset_index()
df = df.replace('NaN', 0)

csv_path = sampling_path / ("sampled.csv")
df.to_csv(csv_path)

df = df.rename(columns={'lat': 'latitude [deg]', })
df = df.rename(columns={'lon': 'longitude [deg]', })
df = df.rename(columns={'depth': 'depth [m]', })
df['altitude [m]'] = 0
df['roll [deg]'] = 0
df['pitch [deg]'] = 0
df['northing [m]'] = 0
df['easting [m]'] = 0
df['heading [deg]'] = 0
df['timestamp [s]'] = 0

csv_path = sampling_path / ("sampled_images.csv")
df.to_csv(csv_path)

# report
res1 = df1.groupby(['groundtruth']).size()
res1_sample = df1_sample.groupby(['groundtruth']).size()

res2 = df2.groupby(['groundtruth']).size()
res2_sample = df2_sample.groupby(['groundtruth']).size()


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




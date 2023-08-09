import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import os
import shutil
from tqdm import tqdm
from PIL import Image
from sampling_tools import sampling_uniform, sampling_stratified, sampling_fixed_number
from sampling_tools import load_uvp5, load_uvp6, load_uvp6_from_csv, copy_image_from_df







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


# create sampling output
time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
output_folder = Path(dir_output)
sampling_path = output_folder / ("sampling" + time_str)
# if not sampling_path.exists():
sampling_path.mkdir(parents=True, exist_ok=True)
sampling_path_images = sampling_path / ("output")
sampling_path_images.mkdir(parents=True, exist_ok=True)

# Loop through the image paths and copy the images to the target directory
copy_image_from_df(df1_sample, sampling_path_images, target_size, cutting_ruler=True, invert_img=True)
copy_image_from_df(df2_sample, sampling_path_images, target_size, cutting_ruler=False, invert_img=False)


# merge two dataframe
df = pd.concat([df1_sample, df2_sample])


# shtulle and remove redundant columns
df = df.drop('path', axis=1)
df = df.sample(frac=1, random_state=np.random.seed())
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.reset_index()
df = df.replace('NaN', 0)

selected_columns = df[['index', 'profile_id', 'object_id', 'depth', 'lat', 'lon', 'datetime',
                       'uvp_model', 'groundtruth', 'relative_path']]

csv_path = sampling_path / ("sampled_images.csv")
selected_columns.to_csv(csv_path)

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


if __name__ == "__main__":
    dir_uvp5 = r'D:\mojmas\files\data\UVP5_images_dataset'
    # dir_uvp6 = r'D:\mojmas\files\data\UVP6Net'
    dir_uvp6 = r'D:\mojmas\files\data\csv_history\sampled_images5.csv'
    dir_output = r'D:\mojmas\files\data\result_sampling'

    which_uvp = 0  # '0' uvp5, '1' uvp6, '2' both uvp merge
    class_type = 0  # '0' means 13 class or '1' means 25 classes

    stratified_flag = False
    sampling_percent_uvp5 = 0.0051  # all=9,884,798
    sampling_percent_uvp6 = 0.08  # all=634,459

    uvp5_cut_flag = False

    sample_fixed = True
    if sample_fixed:
        sampling_percent_uvp5 = 10
        sampling_percent_uvp6 = 10

    target_size = [227, 227]


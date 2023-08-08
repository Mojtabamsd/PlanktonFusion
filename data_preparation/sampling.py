import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pathlib import Path
import datetime
import shutil
from tqdm import tqdm
from PIL import Image


def sampling_fixed_number(df, fixed_number):
    # shffle data
    df = df.sample(frac=1, random_state=np.random.seed())
    sampled_df = df.groupby('groundtruth', group_keys=False).apply(
        lambda x: x.sample(n=fixed_number, replace=True) if len(x) > fixed_number else x.sample(n=len(x), replace=False))

    return sampled_df


path1 = r'D:\mojmas\files\data\csv_history'
dir_renaming = r'D:\mojmas\files\data\UVP_renaming'
dir_output = r'D:\mojmas\files\data\result_sampling'

sample_fixed = True
if sample_fixed:
    sampling_percent_uvp5 = 100
    sampling_percent_uvp6 = 100


target_size = [227, 227]

csv_path1 = os.path.join(path1, 'sampled_images5.csv')
df1 = pd.read_csv(csv_path1, sep=',').reset_index(drop=True, inplace=False)
df1 = df1.rename(columns={'groundthruth': 'groundtruth', })

df1['groundtruth'] = df1['groundtruth'].str.replace('fi', 'fiber')
df1['groundtruth'] = df1['groundtruth'].str.replace('_', '<')
df1['groundtruth'] = df1['groundtruth'].str.replace('copepoda<eggs', 'copepoda eggs')
df1['groundtruth'] = df1['groundtruth'].str.replace('Hydrozoa', 'Hydrozoa_others')
df1['groundtruth'] = df1['groundtruth'].str.replace('Mollusca', 'Mollusca_others')
df1['groundtruth'] = df1['groundtruth'].str.replace('fiberber', 'detritus')
df1['groundtruth'] = df1['groundtruth'].str.replace('fiberlament', 'detritus')
df1['groundtruth'] = df1['groundtruth'].str.replace('Creseis acicula', 'Creseis')
df1['groundtruth'] = df1['groundtruth'].str.replace('antenna<Calanoida', 'Calanoida')
df1 = df1[df1['groundtruth'] != 'Botrynema']

ren = pd.read_csv((dir_renaming + r'\Sheet1.csv'))
merge_dict = dict(zip(ren['taxon'], ren['regrouped2']))


df1['groundtruth'] = df1['groundtruth'].map(merge_dict).fillna(df1['groundtruth'])

if sample_fixed:
    df1_sample = sampling_fixed_number(df1, sampling_percent_uvp5)

res = df1_sample.groupby(['groundtruth']).size()
print(res)

df1_sample['uvp_model'] = ['UVP6'] * len(df1_sample)
dir1_ = r'D:\mojmas\files\data'
df1['relative_path'] = df1['relative_path'].str.replace(r'\\', '/')
df1_sample['path'] = dir1_ + r'\\' + df1_sample['relative_path'].astype(str)

# create sampling output
time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
output_folder = Path(dir_output)
sampling_path = output_folder / ("sampling" + time_str)
# if not sampling_path.exists():
sampling_path.mkdir(parents=True, exist_ok=True)
sampling_path_images = sampling_path / ("output")
sampling_path_images.mkdir(parents=True, exist_ok=True)

for index, row in tqdm(df1_sample.iterrows()):
    image_path = row['path']
    image_filename = os.path.basename(image_path)
    target_path = os.path.join(sampling_path_images, image_filename)
    img = Image.open(image_path)
    inverted_img = img
    # # invert image
    # img_gray = img.convert("L")
    # img_array = np.array(img_gray)
    # max_value = np.iinfo(img_array.dtype).max
    # inverted_array = max_value - img_array
    # inverted_img = Image.fromarray(inverted_array)
    # resize image
    resized_image = inverted_img.resize((target_size[0], target_size[1]), resample=Image.ANTIALIAS)
    resized_image.save(target_path)


# uvp5_oath = r'D:\mojmas\files\data\result_sampling\sampling20230515120451\sampled_images.csv' #1000 sample
uvp5_path = r'D:\mojmas\files\data\result_sampling\sampling20230515133345\sampled_images.csv'   #100 sample
df2_sample = pd.read_csv(uvp5_path, sep=',').reset_index(drop=True, inplace=False)
df2_sample = df2_sample[df2_sample['uvp_model'] != 'UVP6']



# merge two dataframe
df = pd.concat([df1_sample, df2_sample])

# shtulle and remove redundant columns
df = df.drop('path', axis=1)
df = df.sample(frac=1, random_state=np.random.seed())
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.reset_index()

csv_path = sampling_path / ("sampled_images.csv")
df.to_csv(csv_path)
import os
import shutil


def find_and_copy_train(txt_file, source_folder, target_folder):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # image_paths = [line.split()[0] for line in lines]
    image_paths = [line.split()[0].replace('/', os.sep) for line in lines]

    for image_path in image_paths:
        full_source_path = os.path.join(source_folder, image_path)

        if os.path.exists(full_source_path):
            full_target_path = os.path.join(target_folder, image_path)
            os.makedirs(os.path.dirname(full_target_path), exist_ok=True)
            shutil.copy(full_source_path, full_target_path)
            # print(f'Copied: {full_source_path} to {full_target_path}')
        else:
            print(f'File not found: {full_source_path}')


def find_and_copy_test(txt_file, source_folder, target_folder):
    with open(txt_file, 'r') as f:
        while True:
            line = f.readline()
            if len(line) <= 1:
                break
            line = line.strip('\n')
            line = line.split(' ')[0]
            file_seq = line.split('/')
            pre_holder = file_seq[1]
            basename = file_seq[-1]
            target = target_folder + "\\" + pre_holder
            os.makedirs(target, exist_ok=True)
            source = source_folder + "\\" + basename
            # os.makedirs(target, exist_ok=True)
            if os.path.exists(source):
                shutil.copy(source, target + "\\" + basename)
            else:
                print(f'File not found: {source}')

# Usage
txt_file_path_train = r''
txt_file_path_test = r''
source_folder_path = r''
target_folder_path = r''

find_and_copy_train(txt_file_path_train, source_folder_path, target_folder_path)
find_and_copy_test(txt_file_path_test, source_folder_path, target_folder_path)

import os
import random
import shutil

def rename_files(folder_path, val_ratio, test_ratio,file_ext):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    print (file_list)
    # Calculate the number of files for each split
    total_files = len(file_list)
    num_val = val_ratio
    num_test = test_ratio
    num_train = total_files - num_val - num_test

    # Move and rename files to the corresponding splits
    for i, file_name in enumerate(file_list):
        src_path = os.path.join(folder_path, file_name)
        if i < num_val:
            new_file_name = f'1_{i:04d}.{file_ext}'
        elif i < num_val + num_test:
            new_file_name = f'0_{i - num_val:04d}.{file_ext}'
        else:
            new_file_name = f'2_{i - num_val - num_test:04d}.{file_ext}'

        dest_path = os.path.join(folder_path, new_file_name)
        os.rename(src_path, dest_path)

scener = ["scene0000_00_real","scene0000_00_seg","scene0001_00_real","scene0001_00_seg","scene0002_00_real","scene0002_00_seg","scene0003_00_real","scene0003_00_seg","scene0004_00_real","scene0004_00_seg","scene0005_00_real","scene0005_00_seg",]
tupli = [("pose","txt"),("rgb","png")]
for curr_scene in scener:
    for curr_tup in tupli:
        folder_path = f'data/nsvf/ScanNet/{curr_scene}/{curr_tup[0]}'  # Replace this with the path to your folder
        val_ratio = 100
        test_ratio = 100

        rename_files(folder_path, val_ratio, test_ratio,curr_tup[1])
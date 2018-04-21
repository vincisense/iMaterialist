import os
import shutil
import sys

def copy_samples_to_single_dir(source_path, dest_path):

    for root, dirs, files in os.walk(source_path):

        for file in files:
            print(file)
            shutil.copy(os.path.join(root, file), dest_path)


if __name__ == '__main__':
    source_path = '/home/orion/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/train'
    dest_path = '/home/orion/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/ALL'
    copy_samples_to_single_dir(source_path, dest_path)
'''

Move 100 images from each class in train.
Create training and testing set
This to be used during feature collection for error estimate

label_picturenumber_gs.jpeg   [gray scale 64*64]
label_picturenumber_cgs.jpeg  [Cut and gray scale 64*64]
'''


import os
import shutil
import sys


def separate_dataset(train_path, test_path):

    picture_removed = dict()

    for index in range(1, 129):
        picture_removed[str(index)] = 0

    print(picture_removed)

    for root, dirs, files in os.walk(train_path):

        for file in files:
            extract_label = str(file).split('_')[0]

            if picture_removed[extract_label] >= 101:
                print(str(extract_label) + ' completed ')
                pass

            else:
                if picture_removed[extract_label] > 0:

                    temp = picture_removed[extract_label]
                    picture_removed[extract_label] = temp + 1
                    file_path_train = os.path.join(train_path, file)
                    file_path_test = os.path.join(test_path, file)
                    shutil.move(file_path_train, file_path_test)
                else:
                    picture_removed[extract_label] = 1

if __name__ == '__main__':
    # train_folder = '/home/orion/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/train'
    # test_folder = '/home/orion/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/test'

    train_folder = '/media/orion/306820f3-b14d-4713-9670-5ec03229fcd1/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/train'
    test_folder = '/media/orion/306820f3-b14d-4713-9670-5ec03229fcd1/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/test'

    separate_dataset(train_folder, test_folder)
import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFile
import sys
import imghdr
import pandas as pd

import numpy as np
import shutil
import random

import argparse
import csv

class FileRename(object):

    def rename_files_to_add_labels(self, file_path):

        for root, dirs, files in os.walk(file_path):

            for directory in dirs:

                print(directory)
                files = os.walk(os.path.join(file_path, directory)).__next__()[2]

                for file in files:
                    correct_file_name = str(directory) + '_' + str(file)
                    print(os.path.join(file_path, directory, file))
                    print(os.path.join(file_path, directory, correct_file_name))

                    try:
                        os.rename(os.path.join(file_path, directory, file), os.path.join(file_path, directory, correct_file_name))

                    except IOError as e:
                        print
                        'Bad image: %s' % e



class ImageValid(object):

    def check_if_image_is_valid(self, file_path):

        valid_images = pd.DataFrame(columns=['labels', 'valid', 'invalid', 'valid_image_name', 'invalid_image_name'])
        valid_images_index = 0

        for root, dirs, files in os.walk(file_path):
            for directory in dirs:

                print(directory)
                files = os.walk(os.path.join(file_path, directory)).__next__()[2]

                for file in files:
                    # check if file type is valid and the file is ok
                    file_type = imghdr.what(os.path.join(file_path, directory, file))

                    valid_images.set_value(valid_images_index, 'label', str(directory))

                    if not file_type:
                        #print('file type not supported' + str(file) + '' + str(root))
                        print(os.path.join(file_path, directory, file))
                        print(file_type)
                        sys.exit()

                        valid_images.set_value(valid_images_index, 'invalid', 'invalid_image')
                        valid_images.set_value(valid_images_index, 'invalid_image_name', str(file))
                        valid_images_index += 1

                    if file_type:
                        #print('file type supported' + str(file) + '' + str(root))
                        valid_images.set_value(valid_images_index, 'valid', 'valid_image')
                        valid_images.set_value(valid_images_index, 'valid_image_name', str(file))
                        valid_images_index += 1

        valid_images.to_csv("/home/orion/datasets/iMaterialist_data_new/data/output_images.csv")

    def option1(self, file_path):

        for root, dirs, files in os.walk(file_path):
            for directory in dirs:

                print(directory)
                files = os.walk(os.path.join(file_path, directory)).__next__()[2]

                for file in files:

                    parser = ImageFile.Parser()

                    print(os.path.join(file_path, directory, file))

                    try:
                        while True:
                            file = open(os.path.join(file_path, directory, file), "rb")
                            data = file.read(1024)
                            if not data:
                                break
                            parser.feed(data)

                        print(parser)
                        sys.exit()
                        image = parser.close()
                        image.save(os.path.join(file_path, file))

                    except IOError as e:
                        print
                        'Bad image: %s' % e

    def option2(path):

        image = Image.open(path)
        try:
            image.load()
        except IOError as e:
            print
            'Bad image: %s' % e


class CountImages(object):

    def count_images_in_each_folder(self, file_path):
        # return count of each class and the highest number of samples any class have
        # also return the max class name

        max_samples = 0
        max_samples_class = 0
        min_samples = 100000
        min_samples_class = 0

        file_counter = dict()

        class_samples = pd.DataFrame(columns = ['label', 'samples'])
        class_samples_index = 0

        for root, dirs, files in os.walk(file_path):

            for directory in dirs:

                files = os.walk(os.path.join(file_path, directory)).__next__()[2]

                for file in files:

                    directory = os.path.join(root, directory).split("/")[-1]

                    print(directory)

                    if directory not in file_counter:
                        file_counter[directory] =1
                    else:
                        file_counter[directory] = int(file_counter[directory]) + 1

        # print out how many files in each folder
        for index in range(1, 129):
            for key, values in file_counter.items():
                if str(key) == str(index):
                    print(str(key) + str(' => ') + str(values))

                    class_samples.set_value(class_samples_index, 'label', str(key))
                    class_samples.set_value(class_samples_index, 'samples', str(values))
                    class_samples_index += 1

                    if int(values) > max_samples:
                        max_samples = values
                        max_samples_class = int(key)

                    if int(values) < min_samples:
                        min_samples = values
                        min_samples_class = int(key)

        class_samples.to_csv(os.path.join(file_path, 'class_samples.csv'))

        return file_counter, class_samples, max_samples, max_samples_class, min_samples, min_samples_class

    def balance_datasets(self, file_path, file_counter):

        max = 3898
        min = 320

        file_counter_dict = dict()

        for index in range(0, int(file_counter.shape[0])):
            #print(file_counter.ix[index]['label'], file_counter.ix[index]['samples'])
            file_counter_dict[file_counter.ix[index]['label']] = file_counter.ix[index]['samples']

        print('printing key and values')
        for key, value in file_counter_dict.items():
            print(key, ' => ', value)

        for root, dirs, files in os.walk(file_path):

            for dir in dirs:

                print(os.path.join(file_path, dir))

                print(dir)
                print(str(file_counter_dict[int(dir)]))
                print('max samples in any class is ' + str(max))
                print('samples in current class is ' + str(file_counter_dict[int(dir)]))

                if max > int(file_counter_dict[int(dir)]):

                    more_samples_needed = int(max) - int(file_counter_dict[int(dir)])
                    print('new samples are to be generated ' + str(more_samples_needed))

                    # chose a random number of samples (more_samples_needed)
                    # (a) check resolution
                    # (b) if row == columns, then cut row and columns
                    # (c) else cut row and columns
                    # (d) convert to gray-scale
                    # (e) convert to 64x64

                    dir_path = os.path.join(root, dir)
                    print(dir_path)

                    files = os.listdir(dir_path)

                    print(files)

                    random_samples_chosen = np.random.choice(files, more_samples_needed)
                    print(random_samples_chosen)

                    # for regular files
                    for file in files:
                        imagefilepath = os.path.join(dir_path, file)
                        print(imagefilepath)
                        color = cv2.imread(imagefilepath)
                        # If opencv doesnt works due to unusual image file extension try other library (google check)

                        if color is None:
                            continue
                        elif int(color.size) > 100:
                            img = cv2.cvtColor(color.copy(), cv2.COLOR_BGR2RGB)

                        new_file_name = file.split('.')[0] + '_gs' + '.jpeg'
                        outputpath = os.path.join(dir_path, new_file_name)
                        print(outputpath)
                        print(new_file_name)

                        im_gray = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY))

                        print(type(im_gray))
                        pix = np.array(im_gray)
                        print(pix)
                        print(pix.ndim)
                        print(pix.shape)
                        print(pix.shape[0])
                        print(pix.shape[1])

                        # im_new = np.array(im.copy(), dtype='f')
                        #im_new = Image.fromarray(im)

                        #pix_new = np.array(im_new)
                        #print(pix_new)
                        #print(pix_new.ndim)
                        #print(pix_new.shape)

                        #pix = np.array(pix.copy(), dtype='f')
                        im_new = Image.fromarray(pix)
                        print(im_new)
                        im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)
                        im_gray_size.save(outputpath)


                    # for regular files
                    for file in random_samples_chosen:
                        imagefilepath = os.path.join(dir_path, file)
                        print(imagefilepath)
                        color = cv2.imread(imagefilepath)
                        # If opencv doesnt works due to unusual image file extension try other library (google check)

                        if color is None:
                            continue
                        elif int(color.size) > 100:
                            img = cv2.cvtColor(color.copy(), cv2.COLOR_BGR2RGB)

                        # converted grey scale
                        new_file_name = file.split('.')[0] + '_cgs' + '.jpeg'
                        outputpath = os.path.join(dir_path, new_file_name)
                        print(outputpath)
                        print(new_file_name)

                        im_gray = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY))

                        print(type(im_gray))
                        pix = np.array(im_gray)
                        print(pix)
                        print(pix.ndim)
                        print(pix.shape)
                        print(pix.shape[0])
                        print(pix.shape[1])

                        if pix.shape[0] == pix.shape[1]:
                            #### Here you can clip the image 'im_gray' matrix, just delete any side row or column
                            edge_removed = int(float(pix.shape[0])/ float(8))
                            pix = pix[edge_removed:-edge_removed, edge_removed:-edge_removed]
                        else:
                            edge_removed = int(float(pix.shape[0])/float(pix.shape[1]) * 10)
                            print(edge_removed)
                            pix = pix[edge_removed:-edge_removed, edge_removed:-edge_removed]
                            print(pix.shape)

                        # im_new = np.array(im.copy(), dtype='f')
                        # im_new = Image.fromarray(im)

                        # pix_new = np.array(im_new)
                        # print(pix_new)
                        # print(pix_new.ndim)
                        # print(pix_new.shape)

                        # pix = np.array(pix.copy(), dtype='f')
                        im_new = Image.fromarray(pix)
                        print(im_new)
                        im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)
                        im_gray_size.save(outputpath)

                    for file in files:
                        imagefilepath = os.path.join(dir_path, file)
                        os.remove(imagefilepath)

                else:

                    # for regular files
                    for file in files:
                        imagefilepath = os.path.join(dir_path, file)
                        print(imagefilepath)
                        color = cv2.imread(imagefilepath)
                        # If opencv doesnt works due to unusual image file extension try other library (google check)

                        if color is None:
                            continue
                        elif int(color.size) > 100:
                            img = cv2.cvtColor(color.copy(), cv2.COLOR_BGR2RGB)

                        new_file_name = file.split('.')[0] + '_gs' + '.jpeg'
                        outputpath = os.path.join(dir_path, new_file_name)
                        print(outputpath)
                        print(new_file_name)

                        im_gray = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY))

                        print(type(im_gray))
                        pix = np.array(im_gray)
                        print(pix)
                        print(pix.ndim)
                        print(pix.shape)
                        print(pix.shape[0])
                        print(pix.shape[1])
                        im_new = Image.fromarray(pix)
                        print(im_new)
                        im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)
                        im_gray_size.save(outputpath)

                #os.makedirs(os.path.join("/home/orion/datasets/iMaterialist_data_new/data/train_images_done", dir))
                # copy converted directory to new location
                shutil.copytree(os.path.join(file_path, dir),
                                os.path.join("/home/orion/datasets/iMaterialist_data_new/data/train_images_done", dir))


def main():
    obj = CountImages()

    file_path = "/home/orion/datasets/iMaterialist_data_new/data/train_images"

    # (1) count number of images in each class
    #file_counter, class_samples, max_samples, max_samples_class, min_samples, min_samples_class = \
    #    obj.count_images_in_each_folder("/home/orion/datasets/iMaterialist_data_new/data/train_images")

    #print(max_samples, max_samples_class, min_samples, min_samples_class)
    # 3898 42 320 83

    # (2) check if downloaded images are valid or not
    # checkFile = ImageValid()
    # checkFile.check_if_image_is_valid("/home/orion/datasets/iMaterialist_data_new/data/train_images/")
    # Total images 185264

    # TODO This is done, no need to repeat
    # (3) add labels in front of each image file
    #rename = FileRename()
    #rename.rename_files_to_add_labels("/home/orion/datasets/iMaterialist_data_new/data/train_images/")

    # (4) Cut images and replicate the imbalance samples

    file_counter = pd.DataFrame()

    file_counter = pd.read_csv(os.path.join(file_path, 'class_samples.csv'), index_col=False)

    obj.balance_datasets(file_path, file_counter)

    sys.exit()






    #checkFile.option1("/home/orion/datasets/iMaterialist_data/data/train_images/")



    sys.exit()

    count = 1

    root = []
    dirs = []
    files = []
    root2 = []
    dirs2 = []
    files2 = []

    for root, dirs, files in os.walk("/home/orion/datasets/iMaterialist_data/data/train_images"):
        for dir in dirs:
            os.makedirs(os.path.join("/home/orion/datasets/iMaterialist_data/data_grayscale/train_images", dir))

            for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
                for f in files2:
                    imagefilepath = "/home/orion/datasets/iMaterialist_data/data/train_images" + '/' + dir + "/" + f

                    print(imagefilepath)
                    color = cv2.imread(imagefilepath)
                    # If opencv doesnt works due to unusual image file extension try other library (google check)

                    if color is None:
                        continue
                    elif int(color.size) > 100:
                        img = cv2.cvtColor(color.copy(), cv2.COLOR_BGR2RGB)

                    #outputpath = args.outdir + "/" + dir + '_' + f  # you can change output path in any format
                    outputpath = "/home/orion/datasets/iMaterialist_data/data_grayscale/train_images" + "/" + dir + "/" + 'gs_' + f

                    im_gray = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY))

                    print(type(im_gray))
                    pix = np.array(im_gray)
                    print(pix)
                    print(pix.ndim)
                    print(pix.shape)

                    im = pix[200:-200, 200:-200]

                    #im_new = np.array(im.copy(), dtype='f')
                    im_new = Image.fromarray(im)

                    pix_new = np.array(im_new)
                    print(pix_new)
                    print(pix_new.ndim)
                    print(pix_new.shape)

                    #### Here you can clip the image 'im_gray' matrix, just delete any side row or column
                    im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)
                    im_gray_size.save(outputpath)
                    count += 1


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="use pre-trainned resnet model to classify one image")
    #parser.add_argument('--imgdir', type=str, default='/home/dl-box/Downloads/ck/cohn-kanade', help='input image directory for classification')
    #parser.add_argument('--outdir', type=str, default='/home/dl-box/Downloads/ck/cohn', help='output image directory')
    #args = parser.parse_args()

    main()


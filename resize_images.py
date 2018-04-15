import os
import numpy as np
import cv2
import argparse
from PIL import Image, ImageOps, ImageFile
import sys

import imghdr
import csv

import pandas as pd


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

        valid_images.to_csv("/home/orion/datasets/iMaterialist_data/data/output_images.csv")



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

        max = 0
        max_samples_class = 0

        file_counter = dict()

        for root, dirs, files in os.walk(file_path):
                for file in files:

                    dir = root.split("/")[-1]

                    if dir not in file_counter:
                        file_counter[dir] =1
                    else:
                        file_counter[dir] = int(file_counter[dir]) + 1

        # print out how many files in each folder
        for index in range(1, 129):
            for key, values in file_counter.items():
                if str(key) == str(index):
                    print(str(key) + str(' => ') + str(values))

                    if int(values) > max:
                        max = values
                        max_samples_class = int(key)

        return file_counter, max, max_samples_class


def main():

    checkFile = ImageValid()
    checkFile.check_if_image_is_valid("/home/orion/datasets/iMaterialist_data_new/")

    #checkFile.option1("/home/orion/datasets/iMaterialist_data/data/train_images/")

    sys.exit()

    obj = CountImages()
    file_counter, max, max_samples_class = \
        obj.count_images_in_each_folder("/home/orion/datasets/iMaterialist_data/data/train_images")


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


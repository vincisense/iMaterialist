import os
import cv2
from PIL import Image, ImageOps, ImageFile
import sys
import imghdr
import pandas as pd
import shutil
import numpy as np
import argparse
import csv
import random



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



def main():




if __name__ == '__main__':
    main()
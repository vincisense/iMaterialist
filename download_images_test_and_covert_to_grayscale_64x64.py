'''

https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/data
All the data described below are txt files in JSON format.
Overview

test.json: images of which the participants need to generate predictions. Only image URLs are provided.

image{
"image_id" : int,
"url": [string]
}

'''


import json
import requests
import os
import cv2
from PIL import Image, ImageOps, ImageFile
import sys
import imghdr
import shutil
import numpy as np
import argparse
import csv
import random



import urllib.request
import urllib3
from pprint import pprint
from request_retry import requests_retry_session

os.chdir("/media/orion/306820f3-b14d-4713-9670-5ec03229fcd1/datasets/iMaterialist_last/data")

#requests.adapters.DEFAULT_RETRIES = 5


class Data(object):

    def __init__(self):
        self

    def read_input_data_file(self):

        # get current path
        file_path = os.getcwd()
        test_json = json.load(open("test.json"))

        # https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file

        for index in range(0, 10):
            print('url of image')
            print(test_json['images'][index]['url'])
            print('image id')
            print(test_json['images'][index]['image_id'])

        return test_json, file_path

    def create_test_directories(self, file_path):
        # create test directory to store all test outputs

        try:
            print('directory to be created is ' + os.path.join(file_path, "test_images"))
            os.makedirs(os.path.join(file_path, "test_images"))
        except:
            shutil.rmtree(os.path.join(file_path, "test_images"))
            print('directory creation problem, abort!')
            sys.exit()


    def download_test_data(self, test_json, file_path):

        for index in range(0, len(test_json['images'])):

            try:
                url = test_json['images'][index]['url']

                image_extension = str(url[0]).split('.')[-1]
                image_name = str(test_json['images'][index]['image_id']) + '.' + str(image_extension)

                print(index, image_name, url[0], len(url))
                img_data = requests.get(url[0], timeout=10).content
                with open(os.path.join(file_path, "test_images", str(image_name)), 'wb') as handler:
                    handler.write(img_data)
                handler.close()

            except Exception as e:
                print(e)

    def resize_convert_to_grayscale(self, file_path):

        for root, dirs, files in os.walk(file_path):

            # if it already has max number of files
            # for regular files
            for file in files:
                imagefilepath = os.path.join(root, file)
                print(imagefilepath)
                color = cv2.imread(imagefilepath)
                # If opencv doesnt works due to unusual image file extension try other library (google check)

                if color is None:
                    continue
                elif int(color.size) > 100:
                    img = cv2.cvtColor(color.copy(), cv2.COLOR_BGR2RGB)

                new_file_name = file.split('.')[0] + '_gs' + '.jpeg'
                outputpath = os.path.join(root, new_file_name)
                print(outputpath, new_file_name)
                im_gray = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY))
                pix = np.array(im_gray)
                print(pix, pix.ndim, pix.shape, type(im_gray))
                im_new = Image.fromarray(pix)
                print(im_new)
                im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)
                im_gray_size.save(outputpath)

            #os.makedirs(os.path.join("/home/orion/datasets/iMaterialist_data_new/data/train_images_done", dir))
            # copy converted directory to new location
        shutil.copytree(file_path,
                        "/media/orion/306820f3-b14d-4713-9670-5ec03229fcd1/datasets/iMaterialist_last/data/test_images_done")



def main():

    data_obj = Data()
    test_json, file_path = data_obj.read_input_data_file()

    # TODO
    # Data already downloaded from test
    # https://drive.google.com/drive/folders/1x6VQxlJKzFHYkJGO2d_jGrv4xtW65gOi

    # directory for storing testing data
    #data_obj.create_test_directories(file_path)
    #data_obj.download_test_data(test_json, file_path)

    file_path = os.path.join(file_path, "test_dropbox_download")
    data_obj.resize_convert_to_grayscale(file_path)

if __name__ == '__main__':
    main()



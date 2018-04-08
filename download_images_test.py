'''

https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/data
All the data described below are txt files in JSON format.
Overview

train.json: training data with image urls and labels
validation.json: validation data with the same format as train.json
test.json: images of which the participants need to generate predictions. Only image URLs are provided.
sample_submission_randomlabel.csv: example submission file with random predictions to illustrate the submission file format

Training Data

The training dataset includes images from 128 furniture and home goods classes with one ground truth label for each image.
It includes a total of 194,828 images for training and 6,400 images for validation and 12,800 images for testing.
Train and validation sets have the same format as shown below:

{
"images" : [image],
"annotations" : [annotation],
}

image{
"image_id" : int,
"url": [string]
}

annotation{
"image_id" : int,
"label_id" : int
}
'''

# directory where JSON file is stored
# /Users/mdhaque/Desktop/iMaterial

import os
import json
import sys
import requests
import shutil



import urllib.request
import urllib3
from pprint import pprint
from request_retry import requests_retry_session

os.chdir("/home/orion/datasets/iMaterialist")

#requests.adapters.DEFAULT_RETRIES = 5


class Data(object):

    def __init__(self):
        self

    def read_input_data_file(self):

        # get current path
        file_path = os.getcwd()

        train_json = json.load(open("train.json"))
        test_json = json.load(open("test.json"))
        validation_json = json.load(open("validation.json"))

        # https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file

        for index in range(0, 10):
            print('url of image')
            print(train_json['images'][index]['url'])
            print('image id')
            print(train_json['images'][index]['image_id'])
            print('image id')
            print(train_json['annotations'][index]['image_id'])
            print('image label')
            print(train_json['annotations'][index]['label_id'])

        # unique labels
        labels = set()

        # collect all image id's (labels)
        for index in range(0, len(train_json['annotations'])):
            label_name = train_json['annotations'][index]['label_id']
            labels.add(label_name)

        print(labels)

        return train_json, test_json, validation_json, labels, file_path

    def create_test_directories(self, file_path):
        # create test directory to store all test outputs

        try:
            print('directory to be created is ' + os.path.join(file_path, "data", "test_images"))
            os.makedirs(os.path.join(file_path, "data", "test_images"))
        except:
            shutil.rmtree(os.path.join(file_path, "data", "test_images"))
            print('directory creation problem, abort!')
            sys.exit()

    def create_directories(self, labels, file_path):
        # create directories to store the files

        try:
            for label in labels:
                os.makedirs(os.path.join(file_path, "data", str(label)))
        except:
            print('directory creation problem, abort!')
            sys.exit()

    def download_test_data(self, test_json, file_path):

        for index in range(0, len(test_json['images'])):

            try:
                #image_name = test_json['images'][index]['image_id']
                url = test_json['images'][index]['url']
                # not all images end with jpg (extract image extension from url)
                image_name = str(url[0]).split('/')[-1]

                print(index)
                print(url)
                print(type(url))
                print(url[0])
                print(len(url))

                img_data = requests.get(url[0], timeout=10).content
                with open(os.path.join(file_path, "data", "test_images", str(image_name)), 'wb') as handler:
                    handler.write(img_data)
                handler.close()

            except Exception as e:
                print(e)


def main():

    data_obj = Data()
    train_json, test_json, validation_json, labels, file_path = \
        data_obj.read_input_data_file()

    # directory for storing training data
    #data_obj.create_directories(labels, file_path)

    # directory for storing testing data
    data_obj.create_test_directories(file_path)

    data_obj.download_test_data(test_json, file_path)

    #data_obj.download_test_data(train_json, test_json, validation_json, labels, file_path)

if __name__ == '__main__':
    main()



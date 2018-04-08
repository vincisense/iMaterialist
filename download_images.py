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
import shutil
import urllib.request
import urllib3
import requests
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

    def create_directories(self, labels, file_path):
        # create directories to store the files

        try:
            for label in labels:
                os.makedirs(os.path.join(file_path, "data", str(label)))
        except:
            print('directory creation problem, abort!')
            sys.exit()

    def download_data(self, train_json, test_json, validation_json, labels, file_path):

        s = requests.Session()

        for index in range(0, len(train_json['images'])):

            try:
                label = train_json['annotations'][index]['label_id']
                image_name = train_json['annotations'][index]['image_id']
                image_name = str(image_name) + '.jpg'
                url = train_json['images'][index]['url']
                print(index)
                print(url)
                print(type(url))
                print(url[0])
                print(len(url))

                #response = requests_retry_session().get(url[0], timeout=10)
                #print(response.status_code)

                img_data = requests.get(url[0], timeout=10).content
                with open(os.path.join(file_path, "data", str(label), str(image_name)), 'wb') as handler:
                    handler.write(img_data)

                # delete data directory
                #shutil.rmtree(os.path.join(file_path, "data"))


                #print(response.status_code)

                #s = requests.Session()
                #s.auth = ('user', 'pass')
                #s.headers.update({'x-test': 'true'})

                #response = requests_retry_session(sesssion=s).get(
                #    'https://www.peterbe.com'
                #)


                #img_data = requests.get(url[0]).content
                #with open(os.path.join(file_path, "data", str(label), str(image_name)), 'wb') as handler:
                #    handler.write(img_data)

            except Exception as e:
                print(e)


def main():

    data_obj = Data()
    train_json, test_json, validation_json, labels, file_path = \
        data_obj.read_input_data_file()

    data_obj.create_directories(labels, file_path)
    data_obj.download_data(train_json, test_json, validation_json, labels, file_path)

if __name__ == '__main__':
    main()



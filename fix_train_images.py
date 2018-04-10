'''

Fix the training images

(1) Read the actual image name
(2) replace 1.jpg with actual image name
(3) change filename with label_actual_image_name.actual_extesion
(4) save in a single folder

PN: original file name was not used
   rename function doesnt work with file size large
'''


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
        # unique labels
        labels = set()

        # collect all image id's (labels)
        for index in range(0, len(train_json['annotations'])):
            label_name = train_json['annotations'][index]['label_id']
            labels.add(label_name)

        return train_json, test_json, validation_json, labels, file_path


    def collect_all_file_extensions(self, train_json):

        file_extension_type = dict()

        for index in range(0, len(train_json['images'])):

            index = int(index)

            url = train_json['images'][index]['url']
            image_id = train_json['images'][index]['image_id']
            # not all images end with jpg (extract image extension from url)
            label_id = train_json['annotations'][index]['label_id']

            extension = str(url[0])[-4:]
            print(extension)

            if extension not in file_extension_type.keys():
                file_extension_type[extension] = 1
            else:
                file_extension_type[extension] += 1

        for key,values in file_extension_type.items():
            print('number of times ' + str(key) + ' appeared ' + str(values))







    def replace_with_actual_filename(self, train_json, labels, file_path):
        # Extract name in file and compare with actual filename

        for label in labels:
            # construct directory path
            label_directory = os.path.join(file_path, 'data', 'train_images', str(label))
            print(label_directory)

            for root, dirs, files in os.walk(label_directory):

                for file in files:

                    print(file)
                    index = int(str(file).split('.')[0])

                    url = train_json['images'][index]['url']
                    image_id = train_json['images'][index]['image_id']
                    # not all images end with jpg (extract image extension from url)
                    label_id = train_json['annotations'][index]['label_id']

                    correct_file_name = str(label) + '_' + str(index) + '.' + str(url[0]).split('.')[-1]
                    print(correct_file_name)
                    print(url, image_id, label_id)
                    print(os.path.join(label_directory, file))
                    print(os.path.join(label_directory, correct_file_name))
                    os.rename(os.path.join(label_directory, file), os.path.join(label_directory, correct_file_name))


            '''
            url = train_json['images'][index]['url']
            image_id = train_json['images'][index]['image_id']
            # not all images end with jpg (extract image extension from url)
            label_id = train_json['annotations'][index]['label_id']

            '''

if __name__ == '__main__':
    obj = Data()
    train_json, test_json, validation_json, labels, file_path = obj.read_input_data_file()

    obj.collect_all_file_extensions(train_json)

    #obj.replace_with_actual_filename(train_json, labels, file_path)

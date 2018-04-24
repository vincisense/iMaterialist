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


# check if input images are 64x64

def check_images(file_path, counter):

    for root, dirs, files in os.walk(file_path):

        # for regular files
        for file in files:
            print(file)

            if file.endswith('jpeg'):

                imagefilepath = os.path.join(root, file)
                color = cv2.imread(imagefilepath)

                if color is None:
                    continue
                elif int(color.size) > 100:
                    img = cv2.cvtColor(color.copy(), cv2.COLOR_BGR2RGB)

                # converted grey scale
                outputpath = os.path.join(root, file)
                print(outputpath)

                im_gray = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY))
                pix = np.array(im_gray)
                print(outputpath, pix.shape)

                if pix.shape[0] == 64 and pix.shape[1] == 64:
                    continue
                else:
                    print('file name is not 64*64 ' + str(file))
                    counter += 1

                    # im_new = np.array(im.copy(), dtype='f')
                    im_new = Image.fromarray(pix)
                    pix_new = np.array(im_new)
                    print(pix_new)
                    print(pix_new.ndim)
                    print(pix_new.shape)
                    #### Here you can clip the image 'im_gray' matrix, just delete any side row or column
                    im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)
                    im_gray_size.save(outputpath)

    return counter



def main():
    file_path = '/home/orion/datasets/BEGAN/BEGAN-tensorflow/data/AllLabels/splits/train'
    #file_path = "/home/orion/datasets/BEGAN/BEGAN-tensorflow/data/ALL_Labels"

    counter= 0

    counter = check_images(file_path, counter)
    print(file_path)

    print(counter)



if __name__ == '__main__':
    main()


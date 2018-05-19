import os
import cv2
from PIL import Image, ImageOps, ImageFile
import sys
import imghdr
import pandas as pd
import numpy as np

# Filename conversion done already (label added to each file)
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

    def check_if_image_is_valid(self, file_path, data_type):

        valid_images = pd.DataFrame(columns=['valid', 'invalid', 'valid_image_name', 'invalid_image_name'])
        valid_images_index = 0

        counter_valid_images = 0
        counter_invalid_images = 0

        print(file_path)

        for root, dirs, files in os.walk(file_path):

            for file in files:

                print(file)

                # check if file type is valid and the file is ok
                file_type = imghdr.what(os.path.join(file_path, file))

                if not file_type:
                    print('file type not supported' + str(file) + '' + str(root))
                    print(os.path.join(file_path, file))
                    print(file_type)

                    valid_images.set_value(valid_images_index, 'invalid', 'invalid_image')
                    valid_images.set_value(valid_images_index, 'invalid_image_name', str(file))
                    valid_images_index += 1
                    counter_invalid_images += 1

                if file_type:
                    #print('file type supported' + str(file) + '' + str(root))
                    valid_images.set_value(valid_images_index, 'valid', 'valid_image')
                    valid_images.set_value(valid_images_index, 'valid_image_name', str(file))
                    valid_images_index += 1
                    counter_valid_images += 1

        print('number of counter valid images ' + str(counter_valid_images) + ' number of invalid images '
              + str(counter_invalid_images))

        valid_images.to_csv("/home/orion/datasets/iMaterialist_data/data/' + str(data_type) "
                            "+ '_set_output_images.csv")

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




class ConvertImages(object):

    # convert images to 64X64 and grayscale and store in a different directory

    def convert_images_to_grayscale(self, file_path, output_directory):

        # for regular files

        for root, dirs, files in os.walk(file_path):

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
                outputpath = os.path.join(output_directory, new_file_name)
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
                # im_new = Image.fromarray(im)

                # pix_new = np.array(im_new)
                # print(pix_new)
                # print(pix_new.ndim)
                # print(pix_new.shape)

                # pix = np.array(pix.copy(), dtype='f')
                im_new = Image.fromarray(pix)
                print(im_new)
                im_gray_size = ImageOps.fit(im_new.copy(), (64, 64), Image.ANTIALIAS)

                try:
                    im_gray_size.save(outputpath)

                except IOError:
                    print('coundnt write to file')

def main():

    '''
    file_path = '/home/orion/datasets/iMaterialist_data/data/validation_images'
    output_directory = '/home/orion/datasets/iMaterialist_data/data/validation_images_converted'

    image_validation = ImageValid()
    image_validation.check_if_image_is_valid(file_path)

    conversion = ConvertImages()
    conversion.convert_images_to_grayscale(file_path, output_directory)
    '''

    file_path = '/home/orion/datasets/iMaterialist_data/data/test_images'
    output_directory = '/home/orion/datasets/iMaterialist_data/data/test_images_converted'

    image_validation = ImageValid()
    image_validation.check_if_image_is_valid(file_path, data_type)

    conversion = ConvertImages()
    conversion.convert_images_to_grayscale(file_path, output_directory)


if __name__ == '__main__':
    main()
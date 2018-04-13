'''

Some file didnt change name, manually changing file name in directory

'''

import os

data_dir = "/home/orion/datasets/iMaterialist_data/data/train_images/72"


os.chdir("/home/orion/datasets/iMaterialist_data/data/train_images/72")

for root, dirs, files in os.walk(data_dir):

    for file in files:

        if str(file).startswith('72_'):
            continue
        else:
            os.rename(os.path.join(data_dir, file), os.path.join(data_dir, '72_' + str(file)))




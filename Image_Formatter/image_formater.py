import os
import shutil
import glob
import random
import pickle
import numpy as np

from PIL import Image

# Assets
IMAGE_FOLDER_PATH = '/data/shared/sahan_dabarera/Image_Classification_Challange/Data/Images/test'

PKL_OUTPUT_DIR = '/data/shared/sahan_dabarera/Image_Classification_Challange/Data/PKL_Data'

# Read and Resize Images
def get_image_paths(file_path):
    dir_path = file_path+'/**/'
    file_types = ('*.jpg', '*.jpeg', '*.png')
    files_grabbed = []
    for files in file_types:
        files_grabbed.extend(glob.glob(dir_path+files,recursive=True))
    return files_grabbed

# Shuffle Images
def shuffle_images(file_list):
    file_list.sort()
    random.Random(0).shuffle(file_list)
    return file_list

# Create X variable
def get_X(file_list):
    X = np.array([np.array(Image.open(fname).resize((224,224),Image.LANCZOS)) for fname in file_list], dtype=list)
    X = X/255
    X = X.astype('float32')
    return X

# Create Y variables
def get_Y(file_list):
    # 'Caucasian': 0, 'Indian': 1, 'Woman':0, 'Man':1
    yn = []
    yg = []

    for im_path in file_list:
        if 'Caucasian Man' in im_path:
            yn.append(0)
            yg.append(1)
        elif 'Caucasian Woman' in im_path:
            yn.append(0)
            yg.append(0)
        elif 'Indian Man' in im_path:
            yn.append(1)
            yg.append(1)
        elif 'Indian Woman' in im_path:
            yn.append(1)
            yg.append(0)
        else:
            print("Unknown Image")

    yn = np.array(yn).reshape(len(yn),1).astype('float32')
    yg = np.array(yg).reshape(len(yg),1).astype('float32')

    return [yn, yg]

## **********************ENTRY POINT************************ ##
IMAGE_LIST = shuffle_images(get_image_paths(IMAGE_FOLDER_PATH))

SLICED_IMAGE_LIST = np.array_split(IMAGE_LIST, 10)
OUT_X = []
for lst in SLICED_IMAGE_LIST:
    OUT_X.append(get_X(list(lst)))

X = np.vstack(OUT_X)

[Yn, Yg] = get_Y(IMAGE_LIST)

print("X Shape:",X.shape, "\nYn Shape:",Yn.shape, "\nYg Shape:",Yg.shape)

# Save Files as Pickles
out_dict = {'X_train':X, 'Yn': Yn, 'Yg': Yg}
pickle.dump(out_dict, open(PKL_OUTPUT_DIR+'/Test_Data.pkl','wb'))
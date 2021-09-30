"""
    This Script Split Directory Containing Images into Train and Test Sets 
    and saves them in Seperate Folders for Train and Test Dataset, according to a ratio or number of items

    Images Should be Arranged into Classes with Subdirectories
    Main_Directory
        |-Class_1
            |- img1.jpg
            |- img2.jpg
            |- ........
        |-Class_2
            |- img1.jpg
            |- img2.jpg
            |- ........
        ..............

    Use function "split_files" to start the Script
    parameters:
        main_path:str - path to the main folder containing images
        class_names:list - list of classnames Ex: ['Cat', 'Dog']
        is_ratio:bool - if providing a split_ratio, set this to True
        num_test_items: float/int - if split by ratio, provide the ratio, if split by number of items, 
                          provide the number of test items
        dest_path:str - destination folder path

"""
import os
import random
import shutil

from pathlib import Path
from os import listdir
from os.path import isfile, isdir, join


def get_file_list(file_path:str):
    return [file_path+'/'+f for f in listdir(file_path) if isfile(join(file_path, f))]


def random_split(file_list:list, is_ratio:bool, num_test_items):

    if is_ratio:
        num_item_2_pick = round(len(file_list)*num_test_items)
    else:
        num_item_2_pick = num_test_items

    test_list = random.sample(file_list, num_item_2_pick)
    train_list = list(set(file_list).difference(set(test_list)))

    return [train_list, test_list]


def copy_files(file_list: list, dest_path: str):
    try:
        # Check if Class Folder Exists if Not Create it
        if not isdir(dest_path):
            Path(dest_path).mkdir(parents=True, exist_ok=True)
        # Copy the Files there
        for fl in file_list:
            shutil.copyfile(fl, dest_path+'/'+fl.split('/')[-1])
        return True
    except:
        raise
        return False


# Main Method
def split_files(main_path:str, class_names:list, is_ratio:bool, num_test_items, dest_path):

    # Check Main Folder Exists
    if not isdir(main_path):
        raise NotADirectoryError("Main Image Folder Not Found")

    # Check If Sub Folder Exists
    for cl_name in class_names:
        if not isdir(main_path+'/'+cl_name):
            raise NotADirectoryError("Image Class '{}' Folder Not Found".format(cl_name))
    
    # Check Destination Folder Exists
    if not isdir(dest_path):
        raise NotADirectoryError("Destination Folder Not Found")


    # Start the Program
    for cl_name in class_names:
        file_list = get_file_list(main_path+'/'+cl_name)
        [train_files, test_files] = random_split(file_list, is_ratio, num_test_items)

        train_copy_success = copy_files(train_files, dest_path+'/train/'+cl_name)
        test_copy_success = copy_files(test_files, dest_path+'/test/'+cl_name)

        if train_copy_success and test_copy_success:
            print("Class {} Files Succesfully Copied".format(cl_name))
        else:
            if not train_copy_success:
                print("Issue Occured While Trying to Copy Train Set In Class {}".format(cl_name))
            else:
                print("Issue Occured While Trying to Copy Test Set In Class {}".format(cl_name))

# Script Entry Point
# Assets
main_path = 'sample'
class_names = ['Cat', 'Dog']
is_ratio = False
num_test_items = 2
dest_path = 'dest'

# Calling the Main Function with Above Assets
split_files(main_path, class_names, is_ratio, num_test_items, dest_path)
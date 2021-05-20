"""
Creates datasets of smaller images by randomly sampling full-sized
images. This script will crop images, labels, and numpy arrays for
both the training and validation sets. Ensure that train and validate
folders are properly organized and contain numpy array labels prior
to running this script. If new image, label, or array directories
already exist, newly created files will be appended to them. Notice
this will create data for both the training AND validation set.

TO RUN:
python cropping.py <dataset> <pixel_number> <training_image_number> <validation_image_number> --rename
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from PIL import Image
from shutil import rmtree
from tqdm import tqdm

# Query function for command line input
def query(question):
    valid = {'yes': True, 'ye': True, 'y': True,
             'no': False, 'n': False}
    choice = input(question + '[yes/no]: ').lower()
    if choice in valid.keys():
        return valid[choice]
    else:
        chance = input("Please respond with 'yes' or 'no' ('y' or 'n'): ").lower()
        if chance in valid.keys():
            return valid[chance]
        else:
            print('Exiting...')
            sys.exit(1)


def crop(arguments):
    if not os.path.isdir(os.path.join('./data', arguments.dataset)):
        print("Dataset '%s' Does Not Exist!" % arguments.dataset,
              'Cropping Aborted.', sep='\n')
        sys.exit(1)

    # Training and Validation Directories
    train_directory = './data/%s/train' % arguments.dataset
    val_directory = './data/%s/validate' % arguments.dataset

    # Input Image, Label, and Numpy Array Directories
    train_image_dir = os.path.join(train_directory, 'images')
    train_label_dir = os.path.join(train_directory, 'labels')
    train_array_dir = os.path.join(train_directory, 'labels_npy')

    val_image_dir = os.path.join(val_directory, 'images')
    val_label_dir = os.path.join(val_directory, 'labels')
    val_array_dir = os.path.join(val_directory, 'labels_npy')

    # List of Image, Label, and Numpy Array Names
    train_image_names = [f for f in os.listdir(train_image_dir) if not f[0] == "."]
    train_image_names.sort()
    train_label_names = [f for f in os.listdir(train_label_dir) if not f[0] == "."]
    train_label_names.sort()
    try:
        train_array_names = [f for f in os.listdir(train_array_dir) if not f[0] == "."]
        train_array_names.sort()
    except FileNotFoundError:
        print('Training Array Directory Does Not Exist!')
        sys.exit(1)
    val_image_names = [f for f in os.listdir(val_image_dir) if not f[0] == "."]
    val_image_names.sort()
    val_label_names = [f for f in os.listdir(val_label_dir) if not f[0] == "."]
    val_label_names.sort()
    try:
        val_array_names = [f for f in os.listdir(val_array_dir) if not f[0] == "."]
        val_array_names.sort()
    except FileNotFoundError:
        print('Validation Array Directory Does Not Exist!')
        sys.exit(1)

    # Check if length of Image and Label directories are the same
    if not len(train_image_names) == len(train_label_names) == len(train_array_names):
        print('Length of Image, Label, and Array Training Directories Differ!')
        sys.exit(1)

    if not len(val_image_names) == len(val_label_names) == len(val_array_names):
        print('Length of Image, Label, and Array Validation Directories Differ!')
        sys.exit(1)

    # Parameters for Image Cropping
    pixels = arguments.pixel_num
    train_image_num = arguments.train_num
    val_image_num = arguments.val_num
    out_dir = './data/%s_%dpix%dnum' % (arguments.dataset, pixels, train_image_num)

    # Get Dataset name from user if given rename flag
    if arguments.rename:
        print('Rename flag activated.', 'Default dataset name: %s' % out_dir[7:], '', sep='\n')
        prompt = 'What would you like to name the new dataset? '
        out_dir_name = input(prompt)
        if out_dir_name is not None:
            out_dir = './data/%s' % out_dir_name

    # Check if dataset already exists and ask to continue
    if os.path.isdir(out_dir):
        if not query('Dataset directory already exists. Would you like to continue?'):
            print('Cropping Aborted.')
            sys.exit(1)

    # To time entire process
    start_time = time.time()

    # Output Image, Label, and Numpy Array Directories
    train_im_out_dir = os.path.join(out_dir, 'train/images')
    train_lab_out_dir = os.path.join(out_dir, 'train/labels')
    train_arr_out_dir = os.path.join(out_dir, 'train/labels_npy')

    val_im_out_dir = os.path.join(out_dir, 'validate/images')
    val_lab_out_dir = os.path.join(out_dir, 'validate/labels')
    val_arr_out_dir = os.path.join(out_dir, 'validate/labels_npy')

    # Create output directories if necessary
    if not os.path.isdir(train_im_out_dir):
        os.makedirs(train_im_out_dir)
    if not os.path.isdir(train_lab_out_dir):
        os.makedirs(train_lab_out_dir)
    if not os.path.isdir(train_arr_out_dir):
        os.makedirs(train_arr_out_dir)

    if not os.path.isdir(val_im_out_dir):
        os.makedirs(val_im_out_dir)
    if not os.path.isdir(val_lab_out_dir):
        os.makedirs(val_lab_out_dir)
    if not os.path.isdir(val_arr_out_dir):
        os.makedirs(val_arr_out_dir)

    # Loop through all Training Images, Labels, and Numpy Arrays
    print("Creating Training Images (%d): \n" % train_image_num)
    for num in tqdm(range(train_image_num)):
        #start = time.time() * 1000
        # Generate random number for choosing files from directories
        file_num = random.randint(0, len(train_image_names)) - 1

        # Load Image and Label using PIL, Load Numpy Array
        image = Image.open(os.path.join(train_image_dir, train_image_names[file_num]))
        label = Image.open(os.path.join(train_label_dir, train_label_names[file_num]))
        array = np.load(os.path.join(train_array_dir, train_array_names[file_num]))

        # Check if desired image size is greater than size of the image
        if image.size[0]-pixels < 0 and image.size[1]-pixels < 0:
            print('Cropped Images Must be Smaller than Full-Sized Images!',
                  'Image Size: %s' % str(image.size), sep='\n')
            rmtree(out_dir)
            sys.exit(1)

        # Generate Random Upper-Left Coordinate for Cropping
        x_cord = random.randint(0, image.size[0]-pixels)
        y_cord = random.randint(0, image.size[1]-pixels)

        # Names for Image, Label, and Numpy Array
        image_format = train_image_names[file_num][-4:]
        image_name = train_image_names[file_num][:-4] + '_x%d_y%d%s' % (x_cord, y_cord, image_format)
        label_format = train_label_names[file_num][-4:]
        label_name = image_name[:-4] + '_L%s' % label_format
        array_name = image_name[:-4] + '_L.npy'

        # Cropping and Saving Image and Label
        #print('Cropping Image %s at x = %d and y = %d...' % (train_image_names[file_num][:-4], x_cord, y_cord))
        image.crop((x_cord, y_cord, x_cord+pixels, y_cord+pixels)).save(os.path.join(train_im_out_dir, image_name))
        label.crop((x_cord, y_cord, x_cord + pixels, y_cord + pixels)).save(os.path.join(train_lab_out_dir, label_name))

        # Index Numpy Arrays to Grab Same Area as Image and Label
        if len(array.shape) == 2:
            # Check that arrays are two channel
            array_crop = array[y_cord:y_cord+pixels, x_cord:x_cord+pixels]
            np.save(os.path.join(train_arr_out_dir, array_name), array_crop)
        elif len(array.shape) == 3:
            # If arrays are accidentally three channel, script will save two channel arrays
            array = array[:, :, 0]
            array_crop = array[y_cord:y_cord+pixels, x_cord:x_cord+pixels]
            np.save(os.path.join(train_arr_out_dir, array_name), array_crop)
        else:
            # If neither is the case, numpy arrays may not have been created
            print('Check Numpy Training Arrays!', 'Cropping Aborted.', sep='\n')
            rmtree(out_dir)
            sys.exit(1)
        #end = time.time() * 1000
        #total_time = end - start
        #print('Done! Time Taken: %.3f milliseconds \n' % total_time)
    print("Completed Training Images!\n")

    # Loop through all Validation Images, Labels, and Numpy Arrays
    print("Creating Validation Images (%d): \n" % val_image_num)
    for num in tqdm(range(val_image_num)):
        #start = time.time() * 1000
        # Generate random number for choosing files from directories
        file_num = random.randint(0, len(val_image_names)) - 1

        # Load Image and Label using PIL, Load Numpy Array
        image = Image.open(os.path.join(val_image_dir, val_image_names[file_num]))
        label = Image.open(os.path.join(val_label_dir, val_label_names[file_num]))
        array = np.load(os.path.join(val_array_dir, val_array_names[file_num]))

        # Generate Random Upper-Left Coordinate for Cropping
        x_cord = random.randint(0, image.size[0]-pixels)
        y_cord = random.randint(0, image.size[1]-pixels)

        # Names for Image, Label, and Numpy Array
        image_format = val_image_names[file_num][-4:]
        image_name = val_image_names[file_num][:-4] + '_x%d_y%d%s' % (x_cord, y_cord, image_format)
        label_format = val_label_names[file_num][-4:]
        label_name = image_name[:-4] + '_L%s' % label_format
        array_name = image_name[:-4] + '_L.npy'

        # Cropping and Saving Image and Label
        #print('Cropping Image %s at x = %d and y = %d...' % (val_image_names[file_num][:-4], x_cord, y_cord))
        image.crop((x_cord, y_cord, x_cord+pixels, y_cord+pixels)).save(os.path.join(val_im_out_dir, image_name))
        label.crop((x_cord, y_cord, x_cord + pixels, y_cord + pixels)).save(os.path.join(val_lab_out_dir, label_name))

        # Index Numpy Arrays to Grab Same Area as Image and Label
        if len(array.shape) == 2:
            # Check that arrays are two channel
            array_crop = array[y_cord:y_cord+pixels, x_cord:x_cord+pixels]
            np.save(os.path.join(val_arr_out_dir, array_name), array_crop)
        elif len(array.shape) == 3:
            # If arrays are accidentally three channel, script will save two channel arrays
            array = array[:, :, 0]
            array_crop = array[y_cord:y_cord+pixels, x_cord:x_cord+pixels]
            np.save(os.path.join(val_arr_out_dir, array_name), array_crop)
        else:
            # If neither is the case, numpy arrays may not have been created
            print('Check Numpy Validation Arrays!')
            print('Check Numpy Training Arrays!', 'Cropping Aborted.', sep='\n')
            rmtree(out_dir)
            sys.exit(1)
        #end = time.time() * 1000
        #total_time = end - start
        #print('Done! Time Taken: %.3f milliseconds \n' % total_time)
    print("Completed Validation Images!\n")

    time_taken = time.time() - start_time
    print('Total Time Taken:    %.3f seconds' % time_taken)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creation of Datasets of Smaller Images via Random Cropping')
    parser.add_argument('dataset', help='Name of Dataset from which to Crop Images')
    parser.add_argument('pixel_num', type=int, help='Side Length in Pixels of Images in New Dataset')
    parser.add_argument('train_num', type=int, help='Number of Training Images in New Dataset')
    parser.add_argument('val_num', type=int, help='Number of Validation Images in New Dataset')
    parser.add_argument('-r', '--rename', action='store_true', help='Rename the New Dataset', dest='rename')
    args = parser.parse_args()
    crop(args)

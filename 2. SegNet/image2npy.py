"""
Converts label images, typically in PNG or TIF format, to numpy arrays
that can be accessed by PyTorch for training. This step must be taken
before training. To use the cropping.py script to create additional
datasets, numpy arrays must be present in the full-sized parent dataset.

TO RUN: python image2npy.py <dataset> <subdirectory>
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert(arguments):

    # Defines directory for existing labels and numpy labels to be made. The dataset
    # and image set used must be specified correctly in the command line for this
    # script to operate properly.
    labels_dir = './data/%s/%s/labels' % (arguments.dataset, arguments.subdirectory)
    npy_dir = './data/%s/%s/labels_npy' % (arguments.dataset, arguments.subdirectory)

    # Creates directory for numpy arrays if it does not already exist.
    if not os.path.isdir(npy_dir):
        os.makedirs(npy_dir)

    # Creates list image names from label directory over which to iterate.
    try:
        label_names = [f for f in os.listdir(labels_dir) if not f[0] == "."]
        label_names.sort()
    except FileNotFoundError:
        try:
            os.removedirs(npy_dir)
            print("Directory: '%s/' Does Not Exist!" % labels_dir)
        except OSError:
            print('Check Dataset! Missing Labels.')
        sys.exit(1)

    try:
        class_num = int(input('Enter Number of Label Classes: '))
    except ValueError:
        print('Not an Integer!', 'Conversion Aborted.', sep='\n')
        sys.exit(1)

    # Checks if specified output directory is empty and prompts user to continue if it
    # is not empty
    if len(os.listdir(npy_dir)) != 0:
        print('Numpy Array Directory Not Empty!')
        ans = input('Would you like to continue? ')
        if not (ans.lower() == 'yes' or ans.lower() == 'ye' or ans.lower() == 'y'):
            sys.exit(1)

    # Start timer for the entire process.
    start = time.time()

    # Initialize empty lists.
    class_num_list = []
    pixel_list = []
    pixel_max_list = []
    image_list = []

    # Iterate through all images in labels directory
    for file in tqdm(label_names, desc='Initializing', unit=' files'):
        # Open label as a PIL image and convert it to greyscale. This flattens any image
        # format to a one-channel image.
        label = Image.open(os.path.join(labels_dir, file))
        label = label.convert(mode='L')

        # Create a list of tuples, each containing a PIL image and its file name.
        image_list.append((label, file))

        # Compute the number of unique pixel values in each image and the maximum pixel value.
        # The former gives the number of classes in the label, while the latter provides a
        # value to use in normalizing the labels corresponding array.
        num = len(np.unique(list(label.getdata())))
        pix = np.unique(list(label.getdata()))
        pix_max = max(np.unique(list(label.getdata())))

        # Create a list of these values for every label.
        class_num_list.append(num)
        pixel_list.append(pix)
        pixel_max_list.append(pix_max)

    over_list = []
    for n in range(len(class_num_list)):
        if class_num_list[n] > class_num:
            over_list.append((label_names[n][:-4], pixel_list[n]))

    if len(over_list) > 0:
        print('The Following Labels have More Classes Than the Number Specified!', '', sep='\n')
        for element in over_list:
            print('Image: %s' % element[0])
            print('Unique Pixel Values: %s' % str(element[1]), '', sep='\n')
        ans = input('Would you like to continue? ')
        if not (ans.lower() == 'yes' or ans.lower() == 'ye' or ans.lower() == 'y'):
            sys.exit(1)

    # Compute the maximum pixel value in all images
    pixel_max = max(pixel_max_list)

    # To determine the number of classes, we take an upper percentile in the list of class
    # numbers ranging from 80 to 99 based on the number of images in the dataset. Taking the
    # outright maximum is prone to error due to extraneous pixel values often given in the
    # labels. This attempts to correct that error, but assumes at least 80 percent of labels
    # in the initial dataset have the maximum number of image classes.
    print('', 'Number of Classes:   %d' % class_num, sep='\n')
    print('Maximum Pixel Value: %d' % pixel_max, '', sep='\n')

    # Initialize empty lists.
    array_max = []
    array_dim = []

    # Loops through list of tuples containing PIL image and its file name.
    for image in image_list:
        # Convert the PIL image to an array.
        array = np.asarray(image[0], dtype=np.uint8)

        # Normalize the array by dividing its entries by the quotient of 255 and
        # one less than the number of classes. This yields an array whose values
        # are integers and range from 0 to one less than the number of classes.
        array = array / (pixel_max / (class_num - 1))
        array = array.astype(dtype=np.int8)

        # Keep track of the array's maximum value as well as its number of dimensions.
        array_max.append(np.max(array))
        array_dim.append(array.ndim)
        print('Image: %s' % image[1][:-4])
        print('After Transform...')
        print('Unique Values: %s' % str(np.unique(array)))
        print('Array Dimension: %d' % array.ndim, '', sep='\n')

        # Save each array to the output directory.
        np.save(os.path.join(npy_dir, image[1][:-4] + '.npy'), array)

    # Compute the maximum value and number of dimensions across all arrays.
    array_max = max(array_max)
    array_dim = max(array_dim)
    end = time.time()

    # For easy error detection, print whether or not the script printed two
    # dimensional arrays, as well as whether the arrays were properly normalized.
    print('Done! Time Taken:        %.3f seconds' % (end - start))
    print('Normalized Arrays?       %s' % (array_max == (class_num - 1)))
    print('Correct Array Dimension? %s' % (2 == array_dim))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Label Images to Numpy Arrays')
    parser.add_argument('dataset', help='Name of Dataset in which Conversion will Take Place')
    parser.add_argument('subdirectory', choices=['test', 'train', 'validate'],
                        help='Subdirectory in Dataset whose Labels to Convert to Arrays')
    args = parser.parse_args()
    convert(args)

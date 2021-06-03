"""
This script applies a trained neural network from the 'saved' directory to a set
of raw images in the 'data' directory for segmentation. The trained network is
specified by the dataset and hyperparameter configuration version on which it was
trained. The set of images to be segmented must be present within the 'data'
directory, and only contain images of the same size. Images of differing sizes
or non-image type files will cause segmentation to be aborted. Segmentations will
be automatically saved in the 'data' directory.

TO RUN: python apply.py <training_dataset> <training_version> <test_dataset> --gpu
"""

import torch
import torchvision
import torchvision.transforms.functional as tf
from torch.utils import data
from torch.utils.data import DataLoader

import argparse
import os
import ast
import sys
import time
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from statistics import mean

from config import get_config
from models import model_mappings


class Dataset(data.Dataset):
    # Initializes the Dataset class with three attributes: its root directory, image size,
    # and image transformations. The latter should be only conversion to tensors in this case.
    def __init__(self, root, size, transform_img):
        self.img_root = root
        self.img_names = [f for f in os.listdir(self.img_root) if not f[0] == '.']
        self.size = size
        self.transform_img = transform_img

    # For each item, the class converts the images in the directory to RGB format
    # converts them to tensors.
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_dir = os.path.join(self.img_root, img_name)
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform_img(img)
        return img, img_name

    def __len__(self):
        return len(self.img_names)


def average(root_dir):
    # Initialize empty lists which will contain average and std. dev. of each image tensor.
    avg_list = []
    std_list = []

    # Loop through all images in image directory defined above.
    for pic in tqdm([f for f in os.listdir(root_dir) if not f[0] =='.'], desc='Normalizing....', unit=' images'):
        img = Image.open(os.path.join(root_dir, pic))
        tensor = tf.to_tensor(img)
        avg_list.append(torch.mean(tensor).tolist())
        std_list.append(torch.std(tensor).tolist())

    # Returns list with three equal entries for each channel of the RGB image.
    avg = [round(mean(avg_list), 3)] * 3
    std = [round(mean(std_list), 3)] * 3
    return avg, std


def sizing(directory):

    # Initializes empty list and iterates through each image in the dataset. Loop
    # appends the images name and size in a tuple to the list.
    size_list = []
    for image in tqdm([f for f in os.listdir(directory) if not f[0] == '.'], desc='Initializing...', unit=' images'):
        try:
            image_size = Image.open(os.path.join(directory, image)).size
        except (PermissionError, OSError):
            print('Files in Directory That Cannot Be Opened as Images!',
                  'Check Test Directory: %s' % directory)
            sys.exit(1)
        size_list.append((image, image_size))

    # Checks whether of not subsequent image sizes are equal to that of the first image.
    bool_list = []
    for element in range(len(size_list)):
        bool_list.append(size_list[0][1] == size_list[element][1])

    # If all image sizes are not equal, but the majority are equal to that of the first
    # image, we abort segmentation and print the names and sizes of the images whose size
    # does not match that of the first image.
    if len(set(bool_list)) == 2 and bool_list.count(True) >= (len(bool_list)/2):
        indices = [i for i, e in enumerate(bool_list) if e is False]
        print('Test Images of Different Sizes Detected!',
              'Check the Following Images:', '', sep='\n')
        for index in indices:
            print('Image: %s' % size_list[index][0],
                  'Size: %s' % str(size_list[index][1]), sep='\n')
        sys.exit(1)

    # If all image sizes are not equal and the majority are not equal to that of the first
    # image, we abort segmentation and print the names and sizes of the images whose size
    # matches that of the first image.
    elif len(set(bool_list)) == 2 and bool_list.count(True) < (len(bool_list)/2):
        indices = [i for i, e in enumerate(bool_list) if e is True]
        print('Test Images of Different Sizes Detected!',
              'Check the Following Images:', '', sep='\n')
        for index in indices:
            print('Image: %s' % size_list[index][0],
                  'Size: %s' % str(size_list[index][1]), sep='\n')
        sys.exit(1)

    # If all images in the list are the same size, return the size of the first image.
    else:
        return size_list[0][1]


def apply(arguments):

    # Starts timing the segmentation process.
    start = time.time()

    # Obtains the configuration hyperparameters given the version and training dataset. This is solely
    # for obtaining the correct trained network in the segmentation process.
    config = get_config(arguments.dataset, arguments.version)
    method = config['model']
    model_dir = './saved/%s_%s.pth' % (config['name'], method)
    try:
        # If the GPU flag is included and the device is available, the model will be directed to run
        # on the CUDA-enabled GPU.
        # Pulls the correct model from the saved directory and loads its weights and biases which
        # are stored in the 'model_state_dict'
        if arguments.gpu and torch.cuda.is_available():
            model = model_mappings[method](K=config['n_class']).cuda()
            model.load_state_dict(torch.load(model_dir)['model_state_dict'], strict=False)
        else:
            model = model_mappings[method](K=config['n_class'])
            model.load_state_dict(torch.load(model_dir, map_location='cpu')['model_state_dict'], strict=False)
    except KeyError:
        print('%s model does not exist' % method)
        sys.exit(1)

    # Locates the test images and collects their size. The size function checks that all images in the
    # test directory have the same size. If not, segmentation is aborted and the incorrectly sized
    # images are identified with their size in the terminal.
    test_dir = './data/%s' % arguments.test_set

    image_size = sizing(test_dir)[::-1]

    # Define the average tensor value and its standard deviation by the training dataset if it exists.
    # If not, define these values by the test dataset. The former will ensure that this script gives the
    # same output as evaluation.
    if os.path.isfile('./norms/%s_norm.txt' % config['root'][7:]):
        with open('./norms/%s_norm.txt' % config['root'][7:], mode='r') as file:
            avg = ast.literal_eval(file.readline())
            std = ast.literal_eval(file.readline())
    else:
        try:
            avg, std = average(os.path.join(config['root'], 'train/images'))
        except FileNotFoundError:
            print('Neither Training Dataset, nor Normalization File are Present!', '', sep='\n')
            ans = input('Would you like to Normalize by the Incoming Raw Data? ')
            if ans.lower() == 'yes' or ans.lower() == 'ye' or ans.lower() == 'y':
                avg, std = average(test_dir)
            else:
                print('', 'Segmentation Aborted!', sep='\n')
                sys.exit(1)

    # Defines the dataset class for the images to be segmented, which is then loaded into a readable
    # format for PyTorch in the Data loader class below
    test_set = Dataset(root=test_dir, size=image_size,
                       transform_img=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(mean=avg, std=std)
                       ]))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # Gives the directory to which the segmented images are saved. By default, this is within the data
    # directory in a subdirectory determined by the version and architecture used. This is done so as to
    # differentiate segmentations of the same dataset using differently trained networks, as well as keep
    # the test dataset free of directories.
    save_dir = '%s_predictions/%s_%s' % (test_dir, config['name'], method)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # We impose the gradient not to be computed after segmentation since training is not taking place.
    with torch.no_grad():
        model.eval()
        print('Segmentation Underway...', '', sep='\n')

        # We iterate through the dataset, segmenting and saving each image.
        for t, (inputs, names) in enumerate(tqdm(test_loader, desc='Segmenting.....', unit='image')):
            if arguments.gpu and torch.cuda.is_available():
                inputs = inputs.cuda()
            if method == 'pixelnet':
                model.set_train_flag(False)
            outputs = model(inputs)
            predictions = outputs.cpu().argmax(1)
            for i in range(predictions.shape[0]):
                plt.imsave('%s/%s' % (save_dir, names[i]), predictions[i].squeeze(), cmap='gray')

    # Computes the total time taken as well as the average time taken for each image.
    end = time.time() - start
    print('', 'Image Segmentation Complete!', '', sep='\n')
    print('Average Time per Image: %.3f seconds' % (float(end) / test_set.__len__()))
    print('Total Time Taken:       %s' % time.strftime('%H:%M:%S', time.gmtime(end)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply a Trained Neural Network to New Images')

    # Assigns the first two positional arguments to the dataset on which the network was trained
    # and the configuration version used which uniquely specify the trained network to apply.

    parser.add_argument('dataset', help='Dataset on which Network Trained')
    parser.add_argument('version', help='Version Defined in config.py: [v1, v2, ... ]')

    # Assigns the final positional argument to the set of images to be segmented.
    parser.add_argument('test_set', help='Set of Images to be Segmented')

    # If this flag is included, segmentation will take place on the available GPU device
    parser.add_argument('--gpu', action='store_true', help='Include to Perform Segmentation on CUDA-Enabled GPU')

    # Stores the command line arguments as a class with attributes corresponding to each argument,
    # then calls the application function on this class.
    args = parser.parse_args()
    apply(args)

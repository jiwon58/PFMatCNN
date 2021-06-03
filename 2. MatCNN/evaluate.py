# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
from torchnet.meter import meter

# Other Imports
import argparse
import os
import time
import sys
import cv2
import numba
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Matseg Script Imports
from models import model_mappings
from config import get_config
from main import Dataset
from utils import AverageMeter, get_transform
from features import balance
from image2npy import convert as label2np

# Overwrite ConfusionMeter Class for saving every element
class ConfusionMeter(meter.Meter):
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.conf_element = []
        self.normalized = normalized
        self.k = k
        self.reset()
        #self.reset_element()

    def reset(self):
        #self.conf = []
        self.conf.fill(0)
        
    def reset_element(self):
        self.conf_element = []
        #self.conf.fill(0)

    def calc(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        #conf = bincount_2d.reshape((self.k, self.k))
        return bincount_2d.reshape((self.k, self.k))

    def add(self, predicted, target):
        self.conf += self.calc(predicted, target)

    def add_element(self, predicted, target):
        self.conf_element.append(self.calc(predicted, target))

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(2).clip(min=1e-12)[:, None]
        else:
            return self.conf

    def value_element(self, idx):
        conf = self.conf_element[idx]
        if self.normalized:
            conf = conf.astype(np.float32)
            return conf / conf.sum(2).clip(min=1e-12)[:, None]
        else:
            return conf

def bfscorecv(prediction, label, threshold=2):

    prediction = prediction[0].numpy()
    label = label[0].numpy()

    # Get number of classes in ground truth and prediction
    classes_gt = np.unique(label)
    classes_pr = np.unique(prediction)

    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        #print('Classes are not same! GT:', classes_gt, 'Pred:', classes_pr)
        classes = np.concatenate((classes_gt, classes_pr))
        classes = np.unique(classes)
        classes = np.sort(classes)
    else:
        classes = classes_gt

    # Initialize the results array
    m = np.max(classes)
    bfscores = np.zeros((m+1), dtype=float)
    #area_gt = np.zeros((m+1), dtype=float).fill(np.nan)

    # Iterate over classes
    for target_class in classes:

        gt = label.copy().astype(np.uint8)
        if target_class == 0:
            gt[label != 0] = 0
            gt[label == 0] = 1

        else:
            gt[label != target_class] = 0

        # Find contours using OpenCV-python package
        tmp = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        label_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        labels_contour = [label_contours[i][j][0].tolist() for i in range(len(label_contours)) for j in range(len(label_contours[i]))]

        #labels_contour = np.asarray(labels_contour)

        pr = prediction.copy().astype(np.uint8)
        if target_class == 0:
            pr[prediction != 0] = 0
            pr[prediction == 0] = 1

        else:
            pr[pr != target_class] = 0

        tmp = cv2.findContours(pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        prediction_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        prediction_contour = [prediction_contours[i][j][0].tolist() for i in range(len(prediction_contours)) for j in range(len(prediction_contours[i]))]

        # Calculate precision and recall
        precision, numerator, denominator = calc_precision_recall(
            labels_contour, prediction_contour, threshold)    # Precision
        #print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            prediction_contour, labels_contour, threshold)    # Recall
        #print("\trecall:", denominator, numerator)
        if precision == 0 or recall == 0:
            f1 = np.nan
        else:
            f1 = 2*recall*precision/(recall+precision)
        #print(file, ':', f1)
        bfscores[target_class] = f1

    return bfscores

def calc_precision_recall(contours_a, contours_b, threshold):

    count = 0

    for b in range(len(contours_b)):
        # find the nearest distance
        for a in range(len(contours_a)):
            if distance(contours_a[a][0],contours_b[b][0],contours_a[a][1],contours_b[b][1]) < int(threshold):
                count = count + 1
                break
    if count != 0:
        precision_recall = count/len(contours_b)
    else:
        precision_recall = 0

    return precision_recall, count, len(contours_b)

# Calculate the distance with numba package to speed up
@numba.njit
def distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# Computes metrics relevant to evaluation of image segmentation, including
# precision, recall, and accuracy, as well as class and mean IoU.
def metrics(conf_mat, verbose=True):
    c = conf_mat.shape[0]

    # Ignore dividing by zero error
    np.seterr(divide='ignore', invalid='ignore')

    # Divide diagonal entries of confusion matrix by sum of its columns and
    # rows to respectively obtain precision and recall.
    precision = np.nan_to_num(conf_mat.diagonal()/conf_mat.sum(0))
    recall = conf_mat.diagonal()/conf_mat.sum(1)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Initialize empty array for IoU computation
    IoUs = np.zeros(c)
    union_sum = 0

    # Loop through rows of confusion matrix; divide each diagonal entry by the
    # sum of its row and column (while avoiding double-counting that entry).
    for i in range(c):
        union = conf_mat[i, :].sum()+conf_mat[:, i].sum()-conf_mat[i, i]
        union_sum += union
        IoUs[i] = conf_mat[i, i]/union

    # Accuracy computed by dividing sum of confusion matrix diagonal with
    # the sum of the confusion matrix
    acc = conf_mat.diagonal().sum()/conf_mat.sum()
    meanIoU = IoUs.mean()

    if verbose:
        print('precision:', np.round(precision, 5), precision.mean())
        print('recall:', np.round(recall, 5), recall.mean())
        print('Mean IoU:', np.round(IoUs, 5), IoUs.mean())
        print('F1 Score:', np.round(f1_score, 5), f1_score.mean())
        print()
    return acc, precision, recall, meanIoU, IoUs

def overlay(label_dir, pred_dir, save_dir, class_num):
    # First check if there are an equal number of labels and predictions.
    if len([f for f in os.listdir(label_dir)if not f[0] == "."]) != len([f for f in os.listdir(pred_dir)if not f[0] == "."]):
        print('Label and Prediction Directories Have Different Number of Files!')
        sys.exit(1)

    # Get lists of label names and prediction names.
    label_names = sorted([f for f in os.listdir(label_dir) if not f[0] == "."])
    pred_names = sorted([f for f in os.listdir(pred_dir) if not f[0] == "."])

    # Iterate through list of labels and list of predictions.
    for file_num in range(len(label_names)):
        # Start timing for the creation of each overlay.
        start = time.time()
        print("Creating Overlay for Image '%s'..." % pred_names[file_num][:-4], '', sep='\n')

        # Load numpy array label as unsigned integer data type.
        # UINT8 corresponds to integer ranging from 0 to 255, as required by RGB image format.
        label_arr = [f for f in os.listdir(label_dir) if not f[0] == "."]
        label_arr.sort()
        label = np.load(os.path.join(label_dir, label_arr[file_num])).astype(dtype=np.uint8)

        # Load prediction into a PIL image object .
        prediction_arr = [f for f in os.listdir(pred_dir) if not f[0] == "."]
        prediction_arr.sort()
        prediction = Image.open(os.path.join(pred_dir, prediction_arr[file_num]))

        # If there are two classes, convert prediction image to numpy array of ones and zeroes corresponding
        # to each class. If there are more than two classes, convert prediction image to numpy array with
        # entries ranging to 0 to (class_num - 1) corresponding to each class. If number of classes provided
        # is less than 2, abort overlay process.
        if class_num == 2:
            # Convert image from current format to 1-bit black and white format.
            prediction = prediction.convert(mode='1')
            # Convert image to array
            pred_array = np.asarray(prediction).astype(dtype=label.dtype)
        elif class_num > 2:
            # Convert image from current format to 8-bit greyscale format
            prediction = prediction.convert(mode='L')
            # Convert image to array with values potentially ranging from 0 to 255.
            pred_array = np.asarray(prediction)

            # Check if prediction array has values ranging up to 255. This attempts to handle how different
            # image types are converted to numpy arrays. If this is the case, divide each entry so that they
            # range from 0 to (class_num - 1). NOTICE: this process may vary based on label format.
            if np.any(pred_array > class_num):
                pred_array = pred_array / (255 / (class_num - 1))
                pred_array = pred_array.astype(dtype=label.dtype)
            else:
                pred_array = pred_array.astype(dtype=label.dtype)
        else:
            print('Number of Classes Provided Invalid!')
            sys.exit(1)

        # Check if the shape of the label array and prediction array differ.
        if label.shape != pred_array.shape:
            print('Dimension of Label and Prediction Differ!')
            sys.exit(1)

        # Create new array by stacking label three times and scale it range up to 255.
        # This is done so we may convert directly to 'RGB' format.
        correct_array = np.dstack((label, label, label))
        correct_array *= int(255 / (class_num - 1))

        # Create image from layered array and convert it to 'RGBA' format. We will need the alpha
        # channel to adjust transparency of the image.
        correct_image = Image.fromarray(correct_array, mode='RGB').convert(mode='RGBA')

        # If there are two classes, create an image of green false positive pixels and an image of
        # pink false negative pixels to overlay on the label image created above.
        if class_num == 2:
            # Determine if if an entry of the prediction array is greater than the corresponding entry
            # of the label array. This indicates a false positive pixel, which will be mapped to a '1'
            # at this position. Non-false positives will be mapped to a '0' elsewhere.
            false_pos_array = np.greater(pred_array, label).astype(dtype=np.uint8)

            # Determine if if an entry of the prediction array is less than the corresponding entry
            # of the label array. This indicates a false negative pixel, which will be mapped to a '1'
            # at this position. Non-false negatives will be mapped to a '0' elsewhere.
            false_neg_array = np.less(pred_array, label).astype(dtype=np.uint8)

            # Scale both the false positive and false negative arrays so to contain entries of 0 and 255.
            false_pos_array *= 255
            false_neg_array *= 255

            # Stack arrays by the following: zero array of correct size, false positive array, zero
            # array, and false positive array. This creates green image where non-false positive pixels
            # are transparent upon converting to 'RGBA' image format.
            false_pos_array = np.dstack((np.zeros_like(false_pos_array), false_pos_array,
                                         np.zeros_like(false_pos_array), false_pos_array))

            # Stack arrays by the following: false negative array, zero array of correct size, false
            # negative array, and false negative array. This creates pink image where non-false negative
            # pixels are transparent upon converting to 'RGBA' image format.
            false_neg_array = np.dstack((false_neg_array, np.zeros_like(false_neg_array),
                                         false_neg_array, false_neg_array))

            # Create both false positive and false negative images in 'RGBA' format
            false_pos_image = Image.fromarray(false_pos_array, mode='RGBA')
            false_neg_image = Image.fromarray(false_neg_array, mode='RGBA')

            # Overlay false positive and false negative image with label image by their alpha channels
            # to retain transparency. This creates finalized overlay image.
            overlay_image = Image.alpha_composite(correct_image, false_pos_image)
            overlay_image = Image.alpha_composite(overlay_image, false_neg_image)

        # If there are more than two classes, create an image of pink misidentified pixels to overlay
        # onto label image.
        else:
            # Determines where prediction array does not equal label array and outputs a '1' at that
            # position. Correctly identified pixels are mapped to '0' elsewhere.
            false_array = np.not_equal(pred_array, label).astype(dtype=np.uint8)

            # Scale array to have entries 0 and 255 for conversion to 'RGBA' format
            false_array *= 255

            # Stack arrays as mentioned above to create pink image whose correctly identified pixels
            # are transparent, and create image in 'RGBA' format from this layered array.
            false_array = np.dstack((false_array, np.zeros_like(false_array), false_array, false_array))
            false_image = Image.fromarray(false_array, mode='RGBA')

            # Overlay this image with the label image by their alpha channels to retain transparency.
            overlay_image = Image.alpha_composite(correct_image, false_image)

        # Save the resulting overlay image to the specified output directory.
        overlay_image.save(os.path.join(save_dir, pred_names[file_num][:-4] + '_overlay.png'))

        # Compute and display total time taken for each overlay, and indicate that it was successfully saved.
        end = time.time() - start
        print("Overlay for Image '%s' Saved!" % pred_names[file_num][:-4])
        print('Time Taken:    %.3f seconds' % end, '', sep='\n')

def evaluate(args):

    # Defines configuration dictionary and network architecture to use
    config = get_config(args.dataset, args.version)
    method = config['model']

    # Defines the loss function. Takes a tensor as argument to initiate class balancing,
    # which can be obtained from the balance script. Uncomment argument below.
    if config['balance'] and args.gpu and torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss(weight=balance(config)).cuda()
    elif config['balance']:
        criterion = nn.CrossEntropyLoss(weight=balance(config))
    elif args.gpu and torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Maps configuration method to network class defined in models.py
    try:
        if args.gpu and torch.cuda.is_available():
            model = model_mappings[method](K=config['n_class']).cuda()
        else:
            model = model_mappings[method](K=config['n_class'])
    except KeyError:
        print('{} model does not exist'.format(method))
        sys.exit(1)
    
    # Load test data into and iterable dataset with no augmentation and verbose metrics
    test_dir = '{}/{}'.format(config['root'], args.test_folder)
    print('Selected Test Directory: {}'.format(test_dir))
    numpy_dir = '{}/labels_npy'.format(test_dir)
    if not os.path.isdir(numpy_dir):
        print('There is no label numpy array.')
        npargs = argparse.Namespace()
        npargs.dataset = args.dataset
        npargs.subdirectory = args.test_folder
        label2np(npargs)
    test_set = Dataset(test_dir, config['size'], *get_transform(config, is_train=False))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # Load desired trained network from saved directory
    model_dir = './saved/{}_{}.pth'.format(config['name'], method)
    if torch.cuda.is_available() and args.gpu:
        model.load_state_dict(torch.load(model_dir,map_location=torch.device('cuda'))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu'))['model_state_dict'])

    # Define directories to which to save predictions and overlays respectively, and create them if necessary
    save_dir = '{}/predictions/{}_{}'.format(test_dir, args.version, method)
    overlay_dir = '{}/overlays/{}_{}'.format(test_dir, args.version, method)
    labels_dir = os.path.join(test_dir, 'labels_npy')
    if not os.path.isdir('{}/predictions'.format(test_dir)):
        os.mkdir('{}/predictions'.format(test_dir))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    losses = AverageMeter('Loss', ':.5f')

    # Define confusion matrix via built in PyTorch functionality
    conf_meter = ConfusionMeter(config['n_class'])
    conf_meter_indiv = ConfusionMeter(config['n_class'])
    conf_mat_dict = dict()
    bf1_list, bf1_dict = [], {}
    # Torch.no_grad() implies that no back-propagation is occurring during evaluation
    with torch.no_grad():
        # Setting model to eval turns off dropout and batch normalization which would
        # obscure the results of testing
        model.eval()
        for t, (inputs, labels, names) in enumerate(tqdm(test_loader)):
            if args.gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda().long()
            else:
                inputs, labels = inputs, labels.long()
            if method == 'pixelnet':
                model.set_train_flag(False)

            # Compute output and loss function
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Save predictions if evaluating the network
            predictions = outputs.cpu().argmax(1)

            for i in range(predictions.shape[0]):
                plt.imsave('{}/{}.png'.format(save_dir, names[i][:-4]), predictions[i].squeeze(), cmap='gray')
                conf_meter.add_element(outputs.permute(0, 2, 3, 1).contiguous().view(-1, config['n_class']), labels.view(-1))
                conf_mat_dict[names[i][:-4]] = conf_meter.value_element(-1)

            # Update loss
            losses.update(loss.item(), inputs.size(0))

            # Update the confusion matrix given outputs and labels
            conf_meter.add(outputs.permute(0, 2, 3, 1).contiguous().view(-1, config['n_class']), labels.view(-1))

            # Calculate BF1 Score using bfscorecv function
            bf1 = bfscorecv(predictions, labels)
            bf1_list.append(bf1)
            bf1_dict[names[i][:-4]] = list(bf1)

        # Output the updated confusion matrix so that it is interpretable by the metrics function
        conf_mat = conf_meter.value()

        # Obtain Accuracy, IoU, Class Accuracies, and Class IoU from metrics function
        acc, precision, recall, meanIoU, IoUs = metrics(conf_mat, verbose=False)

        #print('loss: %.5f, accuracy: %.5f, mIU: %.5f' % (losses.avg, acc, iou))
        classes = len(bf1_list[0])
        bf1_class = [0] * classes
        for i in range(classes):
            bf1_class[i] = np.nanmean([f[i] for f in bf1_list if len(f) == classes])

        metrics_tot = {'Accuracy':acc, 'Avg. IoU':meanIoU, 'Avg. BF1':np.nanmean(bf1_class)}
        for idx in range(len(list(bf1_class))):
            metrics_tot['IoU class {}'.format(idx+1)] = list(IoUs)[idx]
            metrics_tot['BF1 class {}'.format(idx+1)] = list(bf1_class)[idx]
        metrics_element = {}
        for key in conf_mat_dict:
            acc, precision, recall, meanIoU, IoUs = metrics(conf_mat_dict[key], verbose=False)
            metrics_element[key] = {'Acc': acc, 'Avg. IoU':meanIoU, 'Avg. BF1': np.nanmean(bf1_dict[key])}
            for idx in range(len(list(IoUs))):
                metrics_element[key]['BF1 class {}'.format(idx+1)] = bf1_dict[key][idx]
                metrics_element[key]['IoU class {}'.format(idx+1)] = IoUs[idx]
        print('\n')
        if args.neat:
            from pyprnt import prnt
            prnt(metrics_tot,'\n')
        else:
            print(metrics_tot,'\n')

        # Creates overlays if this is specified in the command line
        if os.path.isdir(labels_dir) and args.overlay:
            if not os.path.isdir(overlay_dir):
                os.makedirs(overlay_dir)
            overlay(labels_dir, save_dir, overlay_dir, config['n_class'])

        # Saves metrics if this is specified in the command line
        if args.save:
            df = DataFrame(metrics_element)
            df = df.transpose()
            df.to_csv('{}_Metrics.csv'.format(save_dir))
            print('\nData saved as {}_Metrics.csv'.format(save_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Network Evaluating for Image Segmentation')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('version', help='version defined in config.py (v1, v2, ...)')
    parser.add_argument('--save', action='store_true', help='save the calculated metrics results')
    parser.add_argument('--gpu', action='store_true', help='include to run evaluating on CUDA-enabled GPU')
    parser.add_argument('--test-folder', default='test', help='name of the folder containing test images, optional') 
    parser.add_argument('--overlay', action='store_true', help='create overlays from network predictions')
    parser.add_argument('--neat', action='store_true', help='Print output using neat table form')
    args = parser.parse_args()
    evaluate(args)
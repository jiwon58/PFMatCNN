"""
Provides miscellaneous functionality like computation of metrics
and averages, as well as data augmentation.
"""

import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchnet.meter import meter
from PIL import Image
from statistics import mean
import cv2
import numba
import pandas as pd
from numba import types

## This class is defined for calculating indiviual files
class ConfusionMeter_indiv(meter.Meter):
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
        super(ConfusionMeter_indiv, self).__init__()
        self.conf = []#np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf = []
        #self.conf.fill(0)

    def add(self, predicted, target):
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
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf.append(conf)
        #self.conf += conf


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

class ConfusionMeter_mask(meter.Meter):
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, mask, normalized=False):
        super(ConfusionMeter_mask, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        mask = mask == int(1)
        self.mask = mask.flatten()
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
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
        #predicted = np.multiply(predicted, self.mask)
        target = target.cpu().numpy()
        #target = np.multiply(target, self.mask)

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

        # hack for bincounting 2 arrays together with masking
        x = predicted + self.k * target

        bincount_2d = np.bincount(x[self.mask].astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf


    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

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
        self.conf_ind = []
        self.normalized = normalized
        self.k = k
        self.reset()
        self.reset_indiv()

    def reset(self):
        #self.conf = []
        self.conf.fill(0)
        
    def reset_indiv(self):
        self.conf_ind = []
        #self.conf.fill(0)

    def add_calc(self, predicted, target):
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

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Recorder object stores metrics and updates itself after each epoch
class Recorder(object):
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)


# This function is only used when calling PixelNet
def generate_rand_ind(labels, n_class, n_samples):
    n_samples_avg = int(n_samples/n_class)
    rand_ind = []
    for i in range(n_class):
        positions = np.where(labels.view(1, -1) == i)[1]
        if positions.size == 0:
            continue
        else:
            rand_ind.append(np.random.choice(positions, n_samples_avg))
    rand_ind = np.random.permutation(np.hstack(rand_ind))
    return rand_ind


# Computes the accuracy of training outputs by summing number of pixels
# in an output equal to its label, and dividing by total number of pixels.
def accuracy(predictions, labels):
    correct = predictions.eq(labels.cpu()).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc


# Takes directory of images and outputs mean and standard deviation of their tensors
# This code runs in the utils.py script to supply the mean and standard deviation of
# the images image set for normalization. Normalization is a necessary step in NN
# training as it allows errors in segmentation to be weighted equally.
def average(root_dir):
    # Defines path to training images. Files must be structured in this way.
    image_dir = os.path.join(root_dir, 'train/images/')

    # Initialize empty lists which will contain average and std. dev. of each image tensor.
    avg_list = []
    std_list = []

    # Loop through all images in image directory defined above.
    for pic in [f for f in os.listdir(image_dir) if not f[0] == "."]:
        img = Image.open(image_dir + pic)
        tensor = TF.to_tensor(img)
        tensor = tensor.float()
        avg_list.append(torch.mean(tensor).tolist())
        std_list.append(torch.std(tensor).tolist())

    # Returns list with three equal entries for each channel of the RGB image.
    avg = [round(mean(avg_list), 3)] * 3
    std = [round(mean(std_list), 3)] * 3
    return avg, std


# Performs random transformations to both an image and its label. Includes
# random rotations, vertical and horizontal flips. Normalizes image with
# respect to mean and standard deviation computed by average function.
def get_transform(config, is_train):
    mean, std, is_aug = average(config['root'])[0], average(config['root'])[1], config['aug']

    # Augmentation only occurs during training and can be toggled in configuration
    if is_train and is_aug:
        transform_label = T.Compose([
            T.RandomRotation(45),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])
    else:
        transform_label = T.Compose([
            T.ToTensor()
        ])

    transform_img = T.Compose([
        transform_label,
        T.Normalize(mean=mean, std=std)
    ])
    return transform_img, transform_label


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
    IoU = IoUs.mean()

    # IoU of second class corresponds to that of Somas, which we record.
    class_iou = IoUs[1]
    if verbose:
        print('precision:', np.round(precision, 5), precision.mean())
        print('recall:', np.round(recall, 5), recall.mean())
        print('IoUs:', np.round(IoUs, 5), IoUs.mean())
        print('F1 Score:', np.round(f1_score, 5), f1_score.mean())
        print()
    return acc, IoU, precision, recall, class_iou

#@numba.njit(nogil=True)
def calc_precision_recall(contours_a, contours_b, threshold):

    count = 0

    for b in range(len(contours_b)):
        # find the nearest distance
        for a in range(len(contours_a)):
            #distance = (contours_a[a][0]-contours_b[b][0])**2 + (contours_a[a][1]-contours_b[b][1]) **2
            if distance(contours_a[a][0],contours_b[b][0],contours_a[a][1],contours_b[b][1]) < int(threshold):
                count = count + 1
                break
    if count != 0:
        precision_recall = count/len(contours_b)
    else:
        precision_recall = 0

    return precision_recall, count, len(contours_b)

@numba.njit
def distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def calc_f1(label, prediction, threshold):
    count = 0
    for b in range(len(prediction)):
        # find the nearest distance
        for a in range(len(label)):
            #distance = np.sqrt((label[a][0]-prediction[b][0])**2 + (label[a][1]-prediction[b][1]) **2)
            if distance(label[a][0],prediction[b][0],label[a][1],prediction[b][1]) < threshold:
                count = count + 1
                break
    if count != 0:
        precision = count/len(prediction)
    else:
        precision = 0

    count = 0
    for b in range(len(label)):
        # find the nearest distance
        for a in range(len(prediction)):
            #distance = np.sqrt((prediction[a][0]-label[b][0])**2 + (prediction[a][1]-label[b][1]) **2)
            if distance(prediction[a][0],label[b][0],prediction[a][1],label[b][1]) < threshold:
                count = count + 1
                break
    if count != 0:
        recall = count/len(label)
    else:
        recall = 0

    if precision == 0 or recall == 0:
        f1 = np.nan
    else:
        f1 = 2*recall*precision/(recall+precision)

    return f1, precision, recall

def bfscorecv(prediction, label, mask=False, threshold=2):

    prediction = prediction[0].numpy()
    label = label[0].numpy()

    # Apply a mask for images
    if isinstance(mask, np.ndarray):
        prediction = np.multiply(prediction, mask).astype(np.uint8)
        label = np.multiply(label, mask).astype(np.uint8)

        tmp = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        mask_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        mask_contour = [mask_contours[i][j][0].tolist() for i in range(len(mask_contours)) for j in range(len(mask_contours[i]))]

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
            if isinstance(mask, np.ndarray):
                gt[mask == 0] = 0
        else:
            gt[label != target_class] = 0
        # Find contours using OpenCV-python package
        tmp = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        label_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        labels_contour = [label_contours[i][j][0].tolist() for i in range(len(label_contours)) for j in range(len(label_contours[i]))]
        if isinstance(mask, np.ndarray):
            labels_contour = [x for x in labels_contour if x not in mask_contour]

        #labels_contour = np.asarray(labels_contour)

        pr = prediction.copy().astype(np.uint8)
        if target_class == 0:
            pr[prediction != 0] = 0
            pr[prediction == 0] = 1
            if isinstance(mask, np.ndarray):
                pr[mask == 0] = 0
        else:
            pr[pr != target_class] = 0

        tmp = cv2.findContours(pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        prediction_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        prediction_contour = [prediction_contours[i][j][0].tolist() for i in range(len(prediction_contours)) for j in range(len(prediction_contours[i]))]
        if isinstance(mask, np.ndarray):
            prediction_contour = [x for x in prediction_contour if x not in mask_contour]

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

# Compute BF1 score using predicted image and ground truth
def bf1score(prediction, label, n_class=None, threshold=2,
            mask=False, mask_contour=False):

    if n_class == None:
        n_class = max(np.unique(label)) + 1

    classes = np.arange(n_class)
    bfscores = np.zeros(n_class, dtype=float)
    precisions = np.zeros(n_class, dtype=float)
    recalls = np.zeros(n_class, dtype=float)

    prediction, label = np.asarray(prediction).astype(np.uint8), np.asarray(label).astype(np.uint8)
    if isinstance(mask, np.ndarray):
        prediction = np.multiply(prediction, mask).astype(np.uint8)
        label = np.multiply(label, mask).astype(np.uint8)

    for target_class in classes:

        gt = label.copy()
        # 0 class for background
        if target_class == 0:
            gt[label != 0] = 0
            gt[label == 0] = 1
            if isinstance(mask, np.ndarray):
                gt[mask == 0] = 0
        else:
            # Make other classes 0 value
            gt[label != target_class] = 0

        # Find contours using OpenCV-python package
        tmp = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        label_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        labels_contour = [label_contours[i][j][0].tolist() for i in range(len(label_contours)) for j in range(len(label_contours[i]))]

        # Exclude mask contour points
        if isinstance(mask, np.ndarray):
            labels_contour = [x for x in labels_contour if x not in mask_contour]

        pr = prediction.copy()
        # 0 class for background
        if target_class == 0:
            pr[prediction != 0] = 0
            pr[prediction == 0] = 1
            if isinstance(mask, np.ndarray):
                pr[mask == 0] = 0
        else:
            # Make other classes 0 value
            pr[pr != target_class] = 0

        tmp = cv2.findContours(pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        prediction_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        prediction_contour = [prediction_contours[i][j][0].tolist() for i in range(len(prediction_contours)) for j in range(len(prediction_contours[i]))]
        # Exclude mask contour points
        if isinstance(mask, np.ndarray):
            prediction_contour = [x for x in prediction_contour if x not in mask_contour]

        precision, numerator, denominator = calc_precision_recall(
            np.array(labels_contour), np.array(prediction_contour), np.array(threshold))    # Precision
        #print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            np.array(prediction_contour), np.array(labels_contour), np.array(threshold))    # Recall
        #print("\trecall:", denominator, numerator)
        if precision == 0 or recall == 0:
            f1 = np.nan
        else:
            f1 = 2*recall*precision/(recall+precision)

        # Save values as list form
        bfscores[target_class] = f1
        precisions[target_class] = precision
        recalls[target_class] = recall

    return bfscores, precisions, recalls

# Save dataframes automatically set width
def autowidth_save(writer, dfs):
    for sheetname, df in dfs.items():  # loop through `dict` of dataframes
        df.to_excel(writer, sheet_name=sheetname)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max([
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                ]) + 5  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save()

# Save dictionaries using excel file
def save_xls(mat_tot, mat_ind, n_class, save_dir):
    pre_col = ['precision_class0','precision_class1']
    re_col = ['recall_class0','recall_class1']
    bf_col = ['bf1score_class0','bf1score_class1']
    bpre_col = ['bprecision_class0','bprecision_class1']
    bre_col = ['brecall_class0','brecall_class1']
    if n_class > 2:
        for i in range(2,config['n_class']):
            bf_col.append('bf1score_class'+str(i))
            pre_col.append('precision_class'+str(i))
            re_col.append('recall_class'+str(i))
            bpre_col.append('bprecision_class'+str(i))
            bre_col.append('brecall_class'+str(i))

    df_total = pd.DataFrame(mat_tot, index=['Average'])

    df_ind = pd.DataFrame(mat_ind).T

    df_ind2 = pd.DataFrame(df_ind.BF1_score.tolist(), columns=bf_col, index=df_ind.index)
    df_ind2['BF1_score'] = df_ind2.mean(numeric_only=True, axis =1)

    df_ind3 = pd.DataFrame(df_ind.Bprecision.tolist(), columns=bpre_col, index=df_ind.index)
    df_ind3['Bprecision'] = df_ind3.mean(numeric_only=True, axis =1)

    df_ind4 = pd.DataFrame(df_ind.Brecall.tolist(), columns=bre_col, index=df_ind.index)
    df_ind4['Brecall'] = df_ind4.mean(numeric_only=True, axis =1)

    df_ind5 = pd.DataFrame(df_ind.precision.tolist(), columns=pre_col, index=df_ind.index)
    df_ind5['precision'] = df_ind5.mean(numeric_only=True, axis =1)

    df_ind6 = pd.DataFrame(df_ind.recall.tolist(), columns=re_col, index=df_ind.index)
    df_ind6['recall'] = df_ind6.mean(numeric_only=True, axis =1)

    df_ind = df_ind.drop(['BF1_score', 'Bprecision', 'Brecall', 'precision', 'recall'], axis=1)
    df_ind = pd.concat([df_ind, df_ind2, df_ind5, df_ind6, df_ind3, df_ind4], axis=1)

    save_dir = save_dir + '_results.xlsx'
    writer = pd.ExcelWriter(save_dir, engine = 'xlsxwriter')
    autowidth_save(pd.ExcelWriter(save_dir, engine = 'xlsxwriter'),
                    {'Total':df_total,'Each':df_ind})

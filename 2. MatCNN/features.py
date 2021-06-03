"""
BALANCE: Handles class balancing, which is important for influencing quicker convergence
in imbalanced training data, i.e. background class dominates class of interest.

PLOTTING: Visualizes results from network training. It plots metrics, including Accuracy,
Loss, Mean IoU, Soma Accuracy, and Training Time, as well as lists the hyperparameters used by the
config.py file.

OVERLAY: Overlays neural network output images with their corresponding labels to indicate
errors in the network's predictions. For datasets with two classes, white and black pixels
correspond to true positives and negatives, respectively, while green and pink pixels
correspond to false positives and negatives, respectively. For datasets with multiple classes,
pink pixels indicate incorrectly segmented pixels regardless of class. This aids in visualizing
how well a network performs upon evaluation.
"""

import os
import sys
import time
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch import tensor


# This code is necessary for plotting function to operate on some platforms.
matplotlib.use('Agg')


def balance(config):
    # Define directory of numpy arrays and initiate empty list of arrays.
    label_dir = os.path.join(config['root'], 'train/labels_npy')
    arrays = []

    # Iterate through the array directory to create list of arrays.
    for file in [f for f in os.listdir(label_dir) if not f[0] == '.']:
        arrays.append(np.load(os.path.join(label_dir, file)))

    # Loop through list of arrays, creating a list of dictionaries for each array with
    # keys corresponding to each class and entries that count each array element
    # corresponding to that class.
    dicts = []
    for array in arrays:
        unique, counts = np.unique(array, return_counts=True)
        dicts.append(dict(zip(unique, counts)))

    # Initialize an empty dictionary. Iterate through list of dictionaries, summing the
    # corresponding entries for each one. This creates a dictionary whose keys correspond
    # to the unique values of all array labels, and whose entries count the number of these
    # pixels.
    draft_dict = {}
    for dictionary in dicts:
        for key in dictionary:
            draft_dict[key] = draft_dict.get(key, 0) + dictionary[key]

    # In case the keys in the dictionary do not constitute a full sequence from 0 to one less
    # than the number of classes, fill in the gaps with ones so that these entries are balanced
    # towards zero. This measure is protective and should not be often utilized.
    if len(draft_dict.keys()) < config['n_class']:
        final_dict = {}
        for key in range(config['n_class']):
            final_dict[key] = draft_dict.get(key, 0)
    else:
        final_dict = draft_dict

    # Convert the dictionary entries to a list and append their reciprocals to another list. If
    # there are zero instances of any single class, append zero to this list.
    reciprocals = []
    for item in list(final_dict.items()):
        try:
            reciprocals.append(1 / item[1])
        except ZeroDivisionError:
            reciprocals.append(0)

    # Normalize the list of reciprocals and return its tensor if its length is equal to the
    # number of classes specified in its configuration.
    normalized = [float(i) / sum(reciprocals) for i in reciprocals]
    if len(normalized) == config['n_class']:
        return tensor(normalized)
    else:
        return None


def plotting(record, config, start_time, total_time, directory, epoch):

    # Determines number of epochs as well as iterations per epoch.
    epochs = list(range(1, epoch+1))
    tick = int(np.ceil(epoch / 40.))
    image_dir = config['root'] + '/train' + '/images/'
    iteration = str(len([f for f in os.listdir(image_dir) if not f[0] == '.'])/int(config['batch_size']))

    # Handles time string formatting for both start and total time.
    time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
    start_str = time.strftime('%B %d %Y, %I:%M:%S %p', time.localtime(start_time))

    # The placement of text boxes in the figure is dependent on the units used for the x-axis of the
    # plot. Hence, the placement of the text need depend on the number of epochs used, and we compute
    # an appropriate textbox shift here.
    shift = epoch/15.

    # Determines plot style (ggplot uses white background and grey plot area) and figure dimensions.
    plt.style.use('ggplot')
    fig = plt.figure(figsize=[16, 9])

    # Creates upper subplot. This will depict training, validation and soma accuracy with respect to
    # epoch, which it obtains for the recorder object.
    plt.subplot(211)
    plt.plot(epochs, record['acc_train'], 'bo-', label='Train Accuracy')
    plt.plot(epochs, record['acc_val'], 'ko--', label='Validate Accuracy')
    plt.plot(epochs, record['class_precision'], 'co--', label='Class Accuracy')
    plt.xticks(np.arange(1, epoch+1, tick))
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlabel('Epochs', fontsize=7)
    plt.ylabel('Accuracy (%)', fontsize=7)
    plt.legend(loc=4, facecolor='white')

    # Titles figure, which must occur before lower subplot so that it remains as the top of the figure.
    # We create our text boxes: the first listing the training metric results while the second lists
    # the hyperparameters used. Both have empty lines between each entry and end with two empty lines.
    fig.suptitle(
        'Neural Network Training Progress: %s (%s)' % (config['root'][7:], start_str),
        y=0.9, fontsize=12)
    text_str1 = '\n'.join([
        'Training Accuracy:    %.4f' % max(record['acc_val']), '',
        'Training Loss:    %.4f' % min(record['loss_val']), '',
        'Training Time:    %s' % time_str, '',
        'Mean IoU:    %.4f' % max(record['mean_iou']), '',
        'Class Accuracy:    %.4f' % max(record['class_precision']), '',
        'Class IoU:    %.4f' % max(record['class_iou']),
        '', ''
    ])
    text_str2 = '\n'.join([
        'Epochs:    %d' % epoch, '',
        'Iterations per Epoch:    %s' % iteration, '',
        'Batch Size:    %d' % config['batch_size'], '',
        'Augmentation:    %s' % config['aug'], '',
        'Class Balancing:    %s' % config['balance'], '',
        ('Initial Learning Rate:    %f' % config['lr']).rstrip('0').rstrip('.'), '',
        'Optimizer:    %s' % config['optimizer']
    ])

    # Plots text boxes onto figure. The y-position of these boxes is hard-coded and subject
    # to change upon the addition of new metrics to display.
    plt.text(epoch+shift, 0.9, 'Results:', figure=fig, weight='bold')
    plt.text(epoch+shift, 0.175, text_str1, figure=fig)
    plt.text(epoch+shift, 0.125, 'Parameters:', figure=fig, weight='bold')
    plt.text(epoch+shift, -0.6, text_str2, figure=fig)

    # Generates lower subplot. This depicts training and validation loss, as well as mean IoU.
    plt.subplot(212)
    plt.plot(epochs, record['loss_train'], 'ro-', label='Train Loss')
    plt.plot(epochs, record['loss_val'], 'ko--', label='Validate Loss')
    plt.plot(epochs, record['mean_iou'], 'mo--', label='Mean IoU')
    plt.xticks(np.arange(1, epoch+1, tick))
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlabel('Epochs', fontsize=7)
    plt.ylabel('Loss', fontsize=7)
    plt.legend(loc=1, facecolor='white')

    # Saves figure to the given directory and prints to indicate that a plot was made.
    plt.savefig(directory, bbox_inches='tight')
    print("Training Results Plot Saved.")


def overlay(label_dir, pred_dir, save_dir, class_num, mask=False):
    # First check if there are an equal number of labels and predictions.

    pred_dir = os.path.join(pred_dir,'masked')
    if len([f for f in os.listdir(label_dir) if not f[0] == '.']) != len([f for f in os.listdir(pred_dir) if f[-5:] == 'm.png']):
        print([f for f in os.listdir(label_dir) if not f[0] == '.'])
        print([f for f in os.listdir(pred_dir) if f[-5:] == 'm.png'])
        print('Label and Prediction Directories Have Different Number of Files!')
        sys.exit(1)

    # Get lists of label names and prediction names.
    label_names = sorted([f for f in os.listdir(label_dir) if not f[0] == '.'])
    pred_names = sorted([f for f in os.listdir(pred_dir) if f[-5:] == 'm.png'])

    # Iterate through list of labels and list of predictions.
    for file_num in range(len(label_names)):
        # Start timing for the creation of each overlay.
        start = time.time()
        # print("Creating Overlay for Image '%s'..." % pred_names[file_num][:-4], '', sep='\n')

        # Load numpy array label as unsigned integer data type.
        # UINT8 corresponds to integer ranging from 0 to 255, as required by RGB image format.
        label = np.load(os.path.join(label_dir, sorted([f for f in os.listdir(label_dir) if not f[0] == '.'])[file_num])).astype(dtype=np.uint8)

        # Load prediction into a PIL image object .
        prediction = Image.open(os.path.join(pred_dir, sorted([f for f in os.listdir(pred_dir) if f[-5:] == 'm.png'])[file_num]))

        # If there are two classes, convert prediction image to numpy array of ones and zeroes corresponding
        # to each class. If there are more than two classes, convert prediction image to numpy array with
        # entries ranging to 0 to (class_num - 1) corresponding to each class. If number of classes provided
        # is less than 2, abort overlay process.
        if class_num == 2:
            # Convert image from current format to 1-bit black and white format.
            prediction = prediction.convert(mode='1')
            # Put a mask if there is a mask
            if isinstance(mask, np.ndarray):
                prediction_m = np.multiply(prediction, mask)
                pred_array_m = np.asarray(prediction_m).astype(dtype=label.dtype)
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

            if isinstance(mask, np.ndarray):
                false_pos_array_m = np.greater(pred_array_m, label).astype(dtype=np.uint8)
                false_neg_array_m = np.less(pred_array_m, label).astype(dtype=np.uint8)

                false_pos_array_m *= 255
                false_neg_array_m *= 255

                false_pos_array_m = np.dstack((np.zeros_like(false_pos_array_m), false_pos_array_m,
                                             np.zeros_like(false_pos_array_m), false_pos_array_m))

                false_neg_array_m = np.dstack((false_neg_array_m, np.zeros_like(false_neg_array_m),
                                             false_neg_array_m, false_neg_array_m))

                false_pos_image_m = Image.fromarray(false_pos_array_m, mode='RGBA')
                false_neg_image_m = Image.fromarray(false_neg_array_m, mode='RGBA')

                overlay_image_m = Image.alpha_composite(correct_image, false_pos_image_m)
                overlay_image_m = Image.alpha_composite(overlay_image_m, false_neg_image_m)


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
        if isinstance(mask, np.ndarray):
            if not os.path.exists(os.path.join(save_dir,'masked')):
                os.mkdir(os.path.join(save_dir,'masked'))
            overlay_image_m.save(os.path.join(save_dir, 'masked',pred_names[file_num][:-4] + '_overlay_m.png'))


        # Compute and display total time taken for each overlay, and indicate that it was successfully saved.
        end = time.time() - start
        print("Overlay for Image '%s' Saved!" % pred_names[file_num][:-4])
        print('Time Taken:    %.3f seconds' % end, '', sep='\n')

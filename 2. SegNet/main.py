"""
Primary Neural Network training script. Dataset must be arranged such that
subdirectories train, validate, and test each contain images, labels, and
labels_npy subdirectories. Labels must have the same name as their
corresponding image.

TO TRAIN:    python3 main.py <dataset> train <version> --save --gpu
TO EVALUATE: python3 main.py <dataset> evaluate <version> --test-folder <test_folder> --gpu

For example, dataset may be 'uhcs', version may be 'v3', and test_folder
may be 'test' or 'validate'.
"""

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

# Other Imports
import argparse
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random, json
from pyprnt import prnt
# MatSeg Script Imports
from utils import (AverageMeter, Recorder, generate_rand_ind, accuracy,
get_transform, metrics, ConfusionMeter_indiv, ConfusionMeter_mask, bfscorecv)
from models import model_mappings
from config import get_config
from features import balance, plotting, overlay


# Overwrites Dataset class offered by PyTorch. Herein, we define directories
# containing images and numpy labels, changing the former into RGB format
# that can be processed by the NN.
class Dataset(data.Dataset):
    def __init__(self, root, size, transform_img, transform_label):
        self.img_root = root + '/images/'
        self.label_root = root + '/labels_npy/'
        self.img_names = sorted([f for f in os.listdir(self.img_root) if not f[0] == '.'])
        self.size = size
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_dir = self.img_root + img_name

        # Name of numpy arrays in labels directory
        # Append '_L' here for the CatSpine dataset
        label_dir = '%s%s.npy' % (self.label_root, img_name[:-4] + '_L')

        # Seed the random generator
        seed = np.random.randint(2147483647)

        # Convert the images to RGB format and seed generator prior to
        # generating transformation to ensure its result matches the label
        with open(img_dir, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            random.seed(seed)
            img = self.transform_img(img)
        label = np.load(label_dir).astype(np.int16)
        label = Image.fromarray(label)
        random.seed(seed)
        label = self.transform_label(label).squeeze()
        return img, label, img_name

    def __len__(self):
        return len(self.img_names)


def get_dataloader(config):
    # Obtain image and label transformations according to random seeding above
    # Augmentation occurs when is_train flag is set to True, hence not during validation
    transform_img_train, transform_label_train = get_transform(config, is_train=True)
    transform_img_val, transform_label_val = get_transform(config, is_train=False)

    # Create Dataset from class defined above
    train_set = Dataset(config['root']+'/train', config['size'], transform_img_train, transform_label_train)
    val_set = Dataset(config['root']+'/validate', config['size'], transform_img_val, transform_label_val)

    # DataLoader combines dataset and sampler (if provided) as well as provides iterable over dataset
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=config['shuffle'], num_workers=0, drop_last=False)
    return train_loader, val_loader


# Training is called within main function and repeats for each epoch. Here, model gives
# network architecture, criterion defines loss function, and train_loader gives dataset
def train(config, model, criterion, optimizer, train_loader, method):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')

    # Setting model to train mode allows for dropout and batch normalization during training,
    # which generally leads to better training results
    model.train()

    # Get learning rate from optimizer status
    print('learning rate =', optimizer.param_groups[0]['lr'])

    # Enumerating the training loader through tqdm creates the progress bar
    for t, (inputs, labels, _) in enumerate(tqdm(train_loader)):
        if args.gpu and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda().long()
        else:
            inputs, labels = inputs, labels.long()
        if method == 'pixelnet':
            model.set_train_flag(True)
            rand_ind = generate_rand_ind(labels.cpu(), n_class=config['n_class'], n_samples=2048)
            model.set_rand_ind(rand_ind)
            labels = labels.view(labels.size(0), -1)[:, rand_ind]

        # Compute output and loss through comparison to labels
        outputs = model(inputs)
        predictions = outputs.cpu().argmax(1)
        loss = criterion(outputs, labels)

        # Measure training accuracy and loss
        accs.update(accuracy(predictions, labels), inputs.shape[0])
        losses.update(loss.item(), inputs.size(0))

        # Compute gradient and back-propagate to update network parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('--- training result ---')
    print('loss: %.5f, accuracy: %.5f' % (losses.avg, accs.avg))
    return losses.avg, accs.avg


# Evaluation is called in main function during both training and testing for each epoch.
# Like training, requires configuration, network, and iterable dataset.
def evaluate(config, model, criterion, validation_loader,  method, mask=False, test_flag=False, save_dir=None):
    losses = AverageMeter('Loss', ':.5f')

    # Define confusion matrix via built in PyTorch functionality
    conf_meter = ConfusionMeter(config['n_class'])
    conf_meter_indiv = ConfusionMeter_indiv(config['n_class'])
    if isinstance(mask, np.ndarray):
        conf_meter_mask = ConfusionMeter_mask(config['n_class'], mask)
    conf_mat_dict = {}
    metrics_tot = {}
    f1_mask, precision_mask, recall_mask, f1_dict = [], [], [], {}
    # Torch.no_grad() implies that no back-propagation is occurring during evaluation
    with torch.no_grad():
        # Setting model to eval turns off dropout and batch normalization which would
        # obscure the results of testing
        model.eval()
        for t, (inputs, labels, names) in enumerate(tqdm(validation_loader)):
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

            if test_flag:
                #torch.save(outputs, '{0}/predictions.pt'.format(save_dir))
                for i in range(predictions.shape[0]):
                    plt.imsave('%s/%s.png' % (save_dir, names[i][:-4]), predictions[i].squeeze(), cmap='gray')
                    if not os.path.exists(os.path.join(save_dir,'masked')):
                        os.mkdir(os.path.join(save_dir,'masked'))
                    plt.imsave('%s/masked/%s.png' % (save_dir,names[i][:-4]+'_m'), np.multiply(predictions[i].squeeze(),mask), cmap='gray')
                    torch.save(outputs, '{0}/{1}.pt'.format(save_dir, names[i][:-4]))
                    conf_meter_indiv.add(outputs.permute(0, 2, 3, 1).contiguous().view(-1, config['n_class']), labels.view(-1))
                    #conf_mat_dict_t[names[i][:-4]] = conf_meter.value()
                    conf_mat_dict[names[i][:-4]] = conf_meter_indiv.value()
                    conf_meter_indiv.reset()

            # Update loss
            losses.update(loss.item(), inputs.size(0))

            # Update the confusion matrix given outputs and labels
            conf_meter.add(outputs.permute(0, 2, 3, 1).contiguous().view(-1, config['n_class']), labels.view(-1))

            if isinstance(mask, np.ndarray):
                # Update the masked confusion matrix given outputs and labels
                conf_meter_mask.add(outputs.permute(0, 2, 3, 1).contiguous().view(-1, config['n_class']), labels.view(-1))

            # Calculate BF1 Score using bfscorecv function
            f1 = bfscorecv(predictions, labels, mask)
            f1_mask.append(f1)
            f1_dict[names[i][:-4]] = list(f1)
            #precision_mask.append(precision)
            #recall_mask.append(recall)


        if test_flag:
            print('--- evaluation result ---')
        else:
            print('--- validation result ---')

        # Output the updated confusion matrix so that it is interpretable by the metrics function
        conf_mat = conf_meter.value()
        if isinstance(mask, np.ndarray):
            conf_mat_mask = conf_meter_mask.value()



        # Obtain Accuracy, IoU, Class Accuracies, and Class IoU from metrics function
        acc, iou, precision, recall, class_iou = metrics(conf_mat, verbose=test_flag)
        if isinstance(mask, np.ndarray):
            acc_mask, iou_mask, _, _, class_iou_mask = metrics(conf_mat_mask, verbose=test_flag)

        if not test_flag:
            print('loss: %.5f, accuracy: %.5f, mIU: %.5f' % (losses.avg, acc, iou))
            print('precision:', np.round(precision, 5))
        else:
            print()
            print('loss: %.5f, accuracy: %.5f, mIU: %.5f' % (losses.avg, acc, iou))
            print('loss_mask: %.5f, accuracy_mask: %.5f, mIU_mask: %.5f' % (losses.avg, acc_mask, iou_mask))
            classes = len(f1_mask[0])
            f1_class = [0] * classes
            for i in range(classes):
                f1_class[i] = np.nanmean([f[i] for f in f1_mask if len(f) == classes])
            print('BF1 score_mask:', np.round(f1_class,5) , np.nanmean(f1_class))
        if isinstance(mask, np.ndarray):
            metrics_tot = {'acc_mask':acc_mask, 'iou_mask':iou_mask,'class_iou_mask':class_iou_mask,'BF1 score':np.nanmean(f1_class)} #'precision':list(precision),'recall':list(recall)
        else:
            metrics_tot = {'acc':acc, 'iou':iou, 'precision':list(precision),'recall':list(recall)}
        metrics_indiv = {}
        for key in conf_mat_dict:
            acc_ind, iou_ind, precision_ind, recall_ind, class_iou_ind = metrics(conf_mat_dict[key][0], verbose=False)
            metrics_indiv[key] = {'acc': acc_ind, 'iou':iou_ind, 'precision':list(precision_ind), 'recall':list(recall_ind), 'class_iou':class_iou_ind,
            'BF1':f1_dict[key]}

        # Define second entry of precision vector as Soma accuracy
        class_precision = precision[1]
    return losses.avg, acc, iou, class_precision, class_iou, metrics_tot, metrics_indiv


def main(args):
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
        print('%s model does not exist' % method)
        sys.exit(1)

    if args.mode == 'train':
        # Starts training time to be completed at end of conditional statement
        start = time.time()

        # Defines directory for trained network, training log, and training plot
        # respectively; create these directories in MatSeg if this is not already done.
        model_dir = './saved/%s_%s.pth' % (config['name'], method)
        log_dir = './log/%s_%s.log' % (config['name'], method)
        plot_dir = './plots/%s_%s.png' % (config['name'], method)

        # Obtains iterable data sets from function above
        train_loader, validation_loader = get_dataloader(config)

        # Conditional outlining choice of optimizer; includes hard-coded hyperparameters
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
        else:
            print('cannot found %s optimizer' % config['optimizer'])
            sys.exit(1)

        # Defines dynamic learning rate reduction. Patience defines the number of epochs after
        # which to reduce the LR should training loss not decrease in those epochs.
        scheduler = ReduceLROnPlateau(optimizer, patience=config['patience'])

        # Gives entries in the Recorder object to measure; obtained from evaluate function
        recorder = Recorder(('loss_train', 'acc_train', 'loss_val', 'acc_val', 'mean_iou', 'class_precision', 'class_iou'))
        iou_val_max = 0

        # Iterate through number of epochs
        for epoch in range(1, config['epoch'] + 1):
            print('Epoch %s:' % epoch)
            loss_train, acc_train = train(config, model, criterion, optimizer, train_loader, method=method)
            loss_val, acc_val, iou_val, class_precision, class_iou, _, _= evaluate(config, model, criterion, validation_loader, method=method)

            # Update learning rate scheduler based on training loss
            scheduler.step(loss_train)

            # Update metrics in Recorder object for each epoch
            recorder.update((loss_train, acc_train, loss_val, acc_val, iou_val, class_precision, class_iou))

            # Save model with higher mean IoU
            if iou_val > iou_val_max and args.save:
                torch.save(recorder.record, log_dir)
                torch.save({
                    'epoch': epoch,
                    'version': args.version,
                    'model_state_dict': model.state_dict(),
                }, model_dir)
                print('validation iou improved from %.5f to %.5f. Model Saved.' % (iou_val_max, iou_val))
                iou_val_max = iou_val

            # Stop training if learning rate is reduced three times or (commented out) if validation loss
            # loss does not decrease for 20 epochs. Otherwise, continue training.
            if (optimizer.param_groups[0]['lr'] / config['lr']) <= 1e-3:
                print('Learning Rate Reduced to 1e-3 of Original Value', 'Training Stopped', sep='\n')
                epochs = epoch
                break
            # elif all(recorder['loss_val'][-20:][i] <= recorder['loss_val'][-20:][i+1] for i in range(19)):
            #     print('Loss has not decreased for previous 20 epochs', 'Training Stopped', sep='\n')
            #     epochs = epoch
            #     break
            else:
                epochs = epoch
                continue

        # Obtain time after all epochs, compute total training time, print and plot results
        end = time.time()
        time_taken = end - start
        print(recorder.record)
        plotting(recorder.record, config, start, time_taken, plot_dir, epochs)

    elif args.mode == 'evaluate':
        # Load test data into and iterable dataset with no augmentation and verbose metrics
        test_dir = '%s/%s' % (config['root'], args.test_folder)
        print(test_dir)
        test_set = Dataset(test_dir, config['size'], *get_transform(config, is_train=False))
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        # Load desired trained network from saved directory
        model_dir = './saved/%s_%s.pth' % (config['name'], method)
        if torch.cuda.is_available() and args.gpu:
            model.load_state_dict(torch.load(model_dir,map_location=torch.device('cuda'))['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu'))['model_state_dict'])

        # Define directories to which to save predictions and overlays respectively, and create them if necessary
        save_dir = '%s/predictions/%s_%s' % (test_dir, args.version, method)
        overlay_dir = '%s/overlays/%s_%s' % (test_dir, args.version, method)
        labels_dir = os.path.join(test_dir, 'labels_npy')
        if not os.path.isdir('%s/predictions' % test_dir):
            os.mkdir('%s/predictions' % test_dir)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        N = 852
        X, Y = N, N
        hard_mask = np.tile(int(0), (Y, X))
        #for coord in [[j,i] for i in range(Y) for j in range(X) if np.sqrt((i-Y/2+15)**2+(j-X/2+10)**2) <= 310]:
        for coord in [[j,i] for i in range(Y) for j in range(X) if np.sqrt((i-Y/2+20)**2+(j-X/2+45)**2) <= 340]:
            hard_mask[coord[0], coord[1]] = int(1)
        _, _, _, _, _, metrics_tot, metrics_indiv = evaluate(config, model, criterion, test_loader, method=method, mask= hard_mask, test_flag=True, save_dir=save_dir)

        with open(save_dir+'/total_conf.json', 'w') as fp:
            json.dump(metrics_tot, fp)
        with open(save_dir+'/indiv_conf.json', 'w') as fp:
            json.dump(metrics_indiv, fp)
        prnt(metrics_tot)

        # Creates overlays if this is specified in the command line
        if os.path.isdir(labels_dir) and args.overlay:
            if not os.path.isdir(overlay_dir):
                os.makedirs(overlay_dir)
            overlay(labels_dir, save_dir, overlay_dir, config['n_class'], hard_mask)

    else:
        print('%s mode does not exist' % args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Network Training for Image Segmentation')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('mode', choices=['train', 'evaluate'], help='mode choices: train, validate, test')
    parser.add_argument('version', help='version defined in config.py (v1, v2, ...)')
    parser.add_argument('--save', action='store_true', help='save the trained model')
    parser.add_argument('--gpu', action='store_true', help='include to run training on CUDA-enabled GPU')
    parser.add_argument('--test-folder', default='test', help='name of the folder containing test images')
    parser.add_argument('--overlay', action='store_true', help='create overlays from network predictions')
    args = parser.parse_args()
    main(args)

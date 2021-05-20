"""
Determines Network Training Configurations. The training process uses this
module to obtain hyperparameters from versions defined for each dataset.
This version must be specified upon training, evaluating, and applying the
network. This module must be updated each time the user would like to train
on a new dataset, or with new hyperparameters.
(This is most easily accomplished in a text or code editor, and not from
within the command line.)
"""
import sys


# Generates the default configuration for a given dataset with the following
# hyperparameters. These default values will later be updated by the version
# specified in the get_config function below.
def gen_default(dataset, n_class, size, batch_size=4, lr=1e-4,
                epoch=40):
    default = {
        'root': './data/' + dataset,
        'n_class': n_class,
        'size': size,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'lr': lr,
        'epoch': epoch,
        'aug': False,
        'shuffle': False,
        'patience': epoch,
        'balance': False
    }
    return default

# Master dictionary whose keys correspond to the dataset on which the NN is to be
# trained. Each entry contains a sub-dictionary on versions, each of which defines
# the set of hyperparameters to be used in training. Custom sets of hyperparameters
# can be created by editing versions to change any of the desired entries in the
# default set listed above. Notice there are no longer entries for the mean and
# standard deviation, which is now handled entirely by the average.py script.
config = {
    'PFSegNet': {
        'default': gen_default('PFSegNet', n_class=2, size=(852, 852)),
        'v1': {'model': 'segnet', 'optimizer': 'Adam', 'aug': True, 'batch_size': 4, 'epoch': 100, 'balance': True},
    },
}


# The get_config function takes the default set of hyperparameters given by the gen_default
# function, and updates, or creates new, entries given the version specified.
def get_config(dataset, version):
    try:
        args = config[dataset]['default'].copy()
    except KeyError:
        print('dataset %s does not exist' % dataset)
        sys.exit(1)
    try:
        args.update(config[dataset][version])
    except KeyError:
        print('version %s is not defined' % version)
    args['name'] = dataset + '_' + version
    return args

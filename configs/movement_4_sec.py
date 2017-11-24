import os
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/code/adamw-evaluation/')
import logging
import time
from collections import OrderedDict
from copy import copy

import numpy as np
from numpy.random import RandomState

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_npy_artifact
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.torch_ext.util import confirm_gpu_availability
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from adamweegeval.movement_4_sec import run_4_sec_exp

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': '/home/schirrmr/data/models/adameegeval/4sec-cv-resnet/',
    }]

    train_test_filenames = [
    {
        'train_filename': 'BhNoMoSc1S001R01_ds10_1-12.BBCI.mat',
        'test_filename': 'BhNoMoSc1S001R13_ds10_1-2BBCI.mat',
    },
    {
        'train_filename': 'FaMaMoSc1S001R01_ds10_1-14.BBCI.mat',
        'test_filename': 'FaMaMoSc1S001R15_ds10_1-2BBCI.mat',
    },
    {
        'train_filename': 'FrThMoSc1S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'FrThMoSc1S001R12_ds10_1-2BBCI.mat',
    },
    {
        'train_filename': 'GuJoMoSc01S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'GuJoMoSc01S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'KaUsMoSc1S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'KaUsMoSc1S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'LaKaMoSc1S001R01_ds10_1-9.BBCI.mat',
        'test_filename': 'LaKaMoSc1S001R10_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'LuFiMoSc3S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'LuFiMoSc3S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'MaJaMoSc1S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'MaJaMoSc1S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'MaKiMoSC01S001R01_ds10_1-4.BBCI.mat',
        'test_filename': 'MaKiMoSC01S001R05_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'MaVoMoSc1S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'MaVoMoSc1S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'PiWiMoSc1S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'PiWiMoSc1S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'RoBeMoSc03S001R01_ds10_1-9.BBCI.mat',
        'test_filename': 'RoBeMoSc03S001R10_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'RoScMoSc1S001R01_ds10_1-11.BBCI.mat',
        'test_filename': 'RoScMoSc1S001R12_ds10_1-2BBCI.mat'
    },
    {
        'train_filename': 'StHeMoSc01S001R01_ds10_1-10.BBCI.mat',
        'test_filename': 'StHeMoSc01S001R11_ds10_1-2BBCI.mat'
    },
    ]

    # Final eval params first, other second
    data_split_params = [
    #                         {
    #     'n_folds': None,
    #     'i_test_fold': None,
    #     'test_on_eval_set': True,
    # },
    ] + dictlistprod({
        'n_folds': [10],
        'i_test_fold': list(range(7,10)),
        'test_on_eval_set': [False],
    })


    old_setup_params = [{
        'valid_set_fraction': None,#0.8 for final eval
        'use_validation_set': True,
        'max_increase_epochs': 80,
        'optimizer_name': 'adam',
        'scheduler_name': None,
        'use_norm_constraint': True,
        'weight_decay': 0,
        'init_lr': 1e-3,
    }]

    #adam_new_setup_params = [{
    #
    #}]

    new_setup_params = [{
        'valid_set_fraction': None,#0.8 for final eval
        'use_validation_set': False,
        'max_increase_epochs': None,
        'optimizer_name': 'adamw',
        'scheduler_name': 'cosine',
        'use_norm_constraint': False,
        'weight_decay': 1e-5,
        'init_lr': 1e-3,
    }]

    #lr_weight_decay_params =

    both_setup_params = old_setup_params + new_setup_params

    seed_params = dictlistprod({
        'np_th_seed': [0,]#1,2,3,4
    })

    preproc_params = dictlistprod({
        'low_cut_hz': [4]#0
    })

    stop_params = [{
        'max_epochs': 800,
    }]


    model_params = dictlistprod({
        'model_name': ['resnet']
    })

    debug_params = [{
        'debug': False,
    }]

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        train_test_filenames,
        data_split_params,
        #old_setup_params,
        #new_setup_params,
        both_setup_params,
        preproc_params,
        model_params,
        stop_params,
        seed_params,
        debug_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    if params['test_on_eval_set'] == False:
        params['test_filename'] = None
    params.pop('test_on_eval_set')
    return params


def run(
        ex, train_filename, test_filename, n_folds,
        i_test_fold, valid_set_fraction, use_validation_set,
        low_cut_hz, model_name, optimizer_name, init_lr,
        scheduler_name, use_norm_constraint,
        weight_decay, max_epochs, max_increase_epochs,
        np_th_seed,
        debug):
    kwargs = locals()
    kwargs.pop('ex')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False

    # check that gpu is available -> should lead to crash if gpu not there
    confirm_gpu_availability()

    exp = run_4_sec_exp(**kwargs)
    end_time = time.time()
    last_row = exp.epochs_df.iloc[-1]
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')

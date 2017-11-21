import os

os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/fbcsp/')
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

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/eegconvnet-paper-replication/',
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

    preproc_params = dictlistprod({
        'low_cut_hz': [4]#0
    })

    model_params = dictlistprod({
        'model_name': ['deep'],#, 'shallow']
    })

    seed_params = dictlistprod({
        'np_th_seed': [0,1,2,3,4]
    })

    debug_params = [{
        'debug': False,
    }]


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        train_test_filenames,
        preproc_params,
        model_params,
        seed_params,
        debug_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run_exp(train_filename, test_filename, np_th_seed, low_cut_hz, model_name,
            debug):
    assert model_name in ['shallow', 'deep']
    from braindecode.datasets.bbci import BBCIDataset

    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    train_folder = '/home/schirrmr/data/BBCI-without-last-runs/'
    test_folder = '/home/schirrmr/data/BBCI-only-last-runs/'
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']

    train_loader = BBCIDataset(os.path.join(train_folder,train_filename),
                               load_sensor_names=load_sensor_names)
    test_loader = BBCIDataset(os.path.join(test_folder, test_filename),
                              load_sensor_names=load_sensor_names)


    log.info("Loading train data...")
    train_cnt = train_loader.load()
    log.info("Loading test data...")
    test_cnt = test_loader.load()

    from collections import OrderedDict
    from braindecode.datautil.trial_segment import \
        create_signal_target_from_raw_mne
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def,
                                                  clean_ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def,
                                                 clean_ival)

    train_clean_trial_mask = np.max(np.abs(train_set.X), axis=(1, 2)) < 800
    test_clean_trial_mask = np.max(np.abs(test_set.X), axis=(1, 2)) < 800

    log.info("Train, clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(train_clean_trial_mask),
        len(train_set.X),
        np.mean(train_clean_trial_mask) * 100))

    log.info("Test,  clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(test_clean_trial_mask),
        len(test_set.X),
        np.mean(test_clean_trial_mask) * 100))

    from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
    from braindecode.datautil.signalproc import exponential_running_standardize
    # lets convert to millivolt for numerical stability of next operations
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    train_cnt = train_cnt.pick_channels(C_sensors)
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    log.info("Resampling train...")
    train_cnt = resample_cnt(train_cnt, 250.0)
    log.info("Highpassing train...")
    train_cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, train_cnt.info['sfreq'], filt_order=3, axis=1),
        train_cnt)
    log.info("Standardizing train...")
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.pick_channels(C_sensors)
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    log.info("Resampling test...")
    test_cnt = resample_cnt(test_cnt, 250.0)
    log.info("Highpassing test...")
    test_cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, test_cnt.info['sfreq'], filt_order=3, axis=1),
        test_cnt)
    log.info("Standardizing test...")
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)

    from collections import OrderedDict
    from braindecode.datautil.trial_segment import \
        create_signal_target_from_raw_mne

    ival = [-500, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set.X = train_set.X[train_clean_trial_mask]
    train_set.y = train_set.y[train_clean_trial_mask]
    test_set.X = test_set.X[test_clean_trial_mask]
    test_set.y = test_set.y[test_clean_trial_mask]

    from braindecode.datautil.splitters import split_into_two_sets

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    from braindecode.torch_ext.util import set_random_seeds, np_to_var
    from braindecode.models.deep4 import Deep4Net
    from braindecode.models.util import to_dense_prediction_model
    from torch import nn

    set_random_seeds(seed=np_th_seed, cuda=True)

    n_classes = int(np.max(train_set.y) + 1)
    n_chans = int(train_set.X.shape[1])
    input_time_length = 1000
    if model_name == 'deep':
        # final_dense_length=69 for full trial length at -500,4000
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                         final_conv_length=2).create_network()
    elif model_name == 'shallow':
        model = ShallowFBCSPNet(
            n_chans, n_classes, input_time_length=input_time_length,
            final_conv_length=30).create_network()
    else:
        raise ValueError("Unknown model name {:s}".format(model_name))
    to_dense_prediction_model(model)
    model.cuda()

    out = model(np_to_var(train_set.X[:1, :, :input_time_length, None]).cuda())

    n_preds_per_input = out.cpu().data.numpy().shape[2]

    from torch import optim

    optimizer = optim.Adam(model.parameters())

    import pandas as pd
    from braindecode.experiments.experiment import Experiment
    from braindecode.datautil.iterators import CropsFromTrialsIterator
    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input,
                                       seed=np_th_seed)

    from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
    max_epochs = 800
    if debug:
        max_epochs = 4
    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', 80)])

    from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
    from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
        RuntimeMonitor, CroppedTrialMisclassMonitor

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    import torch.nn.functional as F
    import torch as th
    # change here this cell
    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2),
                                                      targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=True)
    exp.run()

    return exp


def run(
        ex, train_filename, test_filename, np_th_seed,
        low_cut_hz, model_name, debug):
    kwargs = locals()
    kwargs.pop('ex')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False

    # check that gpu is available -> should lead to crash if gpu not there
    confirm_gpu_availability()

    exp = run_exp(**kwargs)
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

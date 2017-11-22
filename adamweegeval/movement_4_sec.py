import logging
import os
from collections import OrderedDict

import numpy as np
from braindecode.datautil.splitters import select_examples, concatenate_sets, split_into_two_sets
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize

from adamweegeval.run_experiment import run_experiment

log = logging.getLogger(__name__)

def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

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
    cnt = cnt.pick_channels(C_sensors)
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)


    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset

def load_train_valid_test(train_filename, test_filename, n_folds, i_test_fold,
                          valid_set_fraction,
                          use_validation_set, low_cut_hz, debug=False):
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    if test_filename is None:
        assert n_folds is not None
        assert i_test_fold is not None
        assert valid_set_fraction is None
    else:
        assert n_folds is None
        assert i_test_fold is None
        if use_validation_set:
            assert valid_set_fraction is not None

    train_folder = '/home/schirrmr/data/BBCI-without-last-runs/'
    log.info("Loading train...")
    full_train_set = load_bbci_data(os.path.join(train_folder, train_filename),
                                    low_cut_hz=low_cut_hz, debug=debug)

    if test_filename is not None:
        test_folder = '/home/schirrmr/data/BBCI-only-last-runs/'
        log.info("Loading test...")
        test_set = load_bbci_data(os.path.join(test_folder, test_filename),
                                  low_cut_hz=low_cut_hz, debug=debug)
        if use_validation_set:
            assert valid_set_fraction is not None
            train_set, valid_set = split_into_two_sets(full_train_set,
                                                       valid_set_fraction)
        else:
            train_set = full_train_set
            valid_set = None

    # Split data
    if n_folds is not None:
        fold_inds = get_balanced_batches(
            len(full_train_set.X), None, shuffle=False, n_batches=n_folds)

        fold_sets = [select_examples(full_train_set, inds) for inds in
                     fold_inds]

        test_set = fold_sets[i_test_fold]
        train_folds = np.arange(n_folds)
        train_folds = np.setdiff1d(train_folds, [i_test_fold])
        if use_validation_set:
            i_valid_fold = (i_test_fold - 1) % n_folds
            train_folds = np.setdiff1d(train_folds, [i_valid_fold])
            valid_set = fold_sets[i_valid_fold]
            assert i_valid_fold not in train_folds
            assert i_test_fold != i_valid_fold
        else:
            valid_set = None

        assert i_test_fold not in train_folds

        train_fold_sets = [fold_sets[i] for i in train_folds]
        train_set = concatenate_sets(train_fold_sets)
        # Some checks
        if valid_set is None:
            assert len(train_set.X) + len(test_set.X) == len(full_train_set.X)
        else:
            assert len(train_set.X) + len(valid_set.X) + len(test_set.X) == len(
                full_train_set.X)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set

def run_4_sec_exp(train_filename, test_filename, n_folds,
                  i_test_fold, valid_set_fraction, use_validation_set,
                  low_cut_hz, model_name, optimizer_name, init_lr,
                  scheduler_name, use_norm_constraint,
                  weight_decay, max_epochs, max_increase_epochs,
                  np_th_seed,
                  debug):
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        n_folds=n_folds,
        i_test_fold=i_test_fold, valid_set_fraction=valid_set_fraction,
        use_validation_set=use_validation_set,
        low_cut_hz=low_cut_hz, debug=debug)
    if debug:
        max_epochs = 4

    return run_experiment(
        train_set, valid_set, test_set,
        model_name, optimizer_name,
        init_lr=init_lr,
        scheduler_name=scheduler_name,
        use_norm_constraint=use_norm_constraint,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        max_increase_epochs=max_increase_epochs,
        np_th_seed=np_th_seed, )
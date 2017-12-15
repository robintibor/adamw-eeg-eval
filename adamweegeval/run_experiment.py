import numpy as np
import torch.nn.functional as F
import torch as th
from torch import optim
import torch.backends.cudnn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or

from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from adamweegeval.optimizers import AdamW
from adamweegeval.schedulers import (ScheduledOptimizer, CosineAnnealing,
                                     CutCosineAnnealing)
from adamweegeval.resnet import EEGResNet


def run_experiment(
        train_set, valid_set, test_set, model_name, optimizer_name,
        init_lr,
        scheduler_name,
        use_norm_constraint, weight_decay,
        restarts,
        max_epochs,
        max_increase_epochs,
        np_th_seed):
    set_random_seeds(np_th_seed, cuda=True)
    #torch.backends.cudnn.benchmark = True# sometimes crashes?
    if valid_set is not None:
        assert max_increase_epochs is not None
    assert (max_epochs is None) != (restarts is None)
    if max_epochs is None:
        max_epochs = np.sum(restarts)
    n_classes = int(np.max(train_set.y) + 1)
    n_chans = int(train_set.X.shape[1])
    input_time_length = 1000
    if model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         input_time_length=input_time_length,
                         final_conv_length=2).create_network()
    elif model_name == 'shallow':
        model = ShallowFBCSPNet(
            n_chans, n_classes, input_time_length=input_time_length,
            final_conv_length=30).create_network()
    elif model_name in ['resnet-he-uniform', 'resnet-he-normal',
                        'resnet-xavier-normal', 'resnet-xavier-uniform']:
        init_name = model_name.lstrip('resnet-')
        from torch.nn import init
        init_fn = {'he-uniform': lambda w: init.kaiming_uniform(w, a=0),
                   'he-normal': lambda w: init.kaiming_normal(w, a=0),
                   'xavier-uniform': lambda w: init.xavier_uniform(w, gain=1),
                   'xavier-normal': lambda w: init.xavier_normal(w, gain=1)}[init_name]
        model = EEGResNet(in_chans=n_chans, n_classes=n_classes,
                          input_time_length=input_time_length,
                          final_pool_length=10, n_first_filters=48,
                          conv_weight_init_fn=init_fn).create_network()
    else:
        raise ValueError("Unknown model name {:s}".format(model_name))
    if 'resnet' not in model_name:
        to_dense_prediction_model(model)
    model.cuda()
    model.eval()

    out = model(np_to_var(train_set.X[:1, :, :input_time_length, None]).cuda())

    n_preds_per_input = out.cpu().data.numpy().shape[2]


    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay,
                               lr=init_lr)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), weight_decay=weight_decay,
                          lr=init_lr)

    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input,
                                       seed=np_th_seed)

    if scheduler_name is not None:
        if scheduler_name == 'cosine':
            n_updates_per_epoch = sum(
                [1 for _ in iterator.get_batches(train_set, shuffle=True)])
            if restarts is None:
                n_updates_per_period = n_updates_per_epoch * max_epochs
            else:
                n_updates_per_period = np.array(restarts) * n_updates_per_epoch
            scheduler = CosineAnnealing(n_updates_per_period)
            optimizer = ScheduledOptimizer(scheduler, optimizer)
        elif scheduler_name == 'cut_cosine':
            # TODO: integrate with if clause before, now just separate
            # to avoid messing with code
            n_updates_per_epoch = sum(
                [1 for _ in iterator.get_batches(train_set, shuffle=True)])
            if restarts is None:
                n_updates_per_period = n_updates_per_epoch * max_epochs
            else:
                n_updates_per_period = np.array(restarts) * n_updates_per_epoch
            scheduler = CutCosineAnnealing(n_updates_per_period)
            optimizer = ScheduledOptimizer(scheduler, optimizer)
        else:
            raise ValueError("Unknown scheduler")
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    if use_norm_constraint:
        model_constraint = MaxNormDefaultConstraint()
    else:
        model_constraint = None
    # change here this cell
    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2),
                                                      targets)

    if valid_set is not None:
        run_after_early_stop = True
        do_early_stop = True
        remember_best_column = 'valid_misclass'
        stop_criterion = Or([MaxEpochs(max_epochs),
                             NoDecrease('valid_misclass', max_increase_epochs)])
    else:
        run_after_early_stop = False
        do_early_stop = False
        remember_best_column = None
        stop_criterion = MaxEpochs(max_epochs)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column=remember_best_column,
                     run_after_early_stop=run_after_early_stop, cuda=True,
                     do_early_stop=do_early_stop)
    exp.run()
    return exp

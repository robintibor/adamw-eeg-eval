from braindecode.torch_ext.util import np_to_var, var_to_np
from torch.optim import SGD
import torch as th
import numpy as np
from adamweegeval.optimizers import AdamW
from adamweegeval.schedulers import CosineAnnealing, ScheduledOptimizer


def test_cosine_annealing_should_affect_update_in_sgd():
    init_w = np.float32(3)
    w_var = np_to_var(init_w, dtype=np.float64)
    x_var = np_to_var(2, dtype=np.float64)
    y_var = np_to_var(100, dtype=np.float64)
    w_var = th.nn.Parameter(w_var.data)
    lr = 0.1
    grad = -2
    optim = ScheduledOptimizer(CosineAnnealing(10), SGD([w_var], lr=lr), )

    n_epochs = 10
    grad_times_lr_per_epoch = grad * lr * (
    0.5 * np.cos(np.pi * np.arange(0, n_epochs) / (n_epochs)) + 0.5)
    for i_epoch in range(n_epochs):
        expected_subtracted_gradient = np.sum(
            grad_times_lr_per_epoch[:i_epoch + 1])
        loss = th.abs(y_var - w_var * x_var)
        optim.zero_grad()
        loss.backward()
        optim.step()
        assert np.allclose(init_w - expected_subtracted_gradient,
                           var_to_np(w_var))


def test_cosine_annealing_should_affect_weight_decay_adamw():
    init_w = np.float32(3)
    w_var = np_to_var(init_w, dtype=np.float64)
    x_var = np_to_var(2, dtype=np.float64)
    y_var = np_to_var(100, dtype=np.float64)
    w_var = th.nn.Parameter(w_var.data)
    wd = 0.1
    lr = 0
    optim = AdamW([w_var], lr=lr, weight_decay=wd)
    optim = ScheduledOptimizer(CosineAnnealing(10), optim)
    n_epochs = 10
    cosine_val_per_epoch = 0.5 * np.cos(
        np.pi * np.arange(0, n_epochs) / (n_epochs)) + 0.5
    decayed_w = init_w
    for i_epoch in range(n_epochs):
        decayed_w = decayed_w * (1 - wd * cosine_val_per_epoch[i_epoch])
        loss = th.abs(y_var - w_var * x_var)
        optim.zero_grad()
        loss.backward()
        optim.step()
        assert np.allclose(decayed_w, var_to_np(w_var))

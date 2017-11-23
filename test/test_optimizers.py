from braindecode.torch_ext.util import np_to_var, var_to_np
from torch.optim import SGD, Adam
import torch as th
import numpy as np
from adamweegeval.optimizers import AdamW


def test_sanity_check_sgd():
    # sanity check SGD
    w_var = np_to_var(3, dtype=np.float32)
    x_var = np_to_var(2, dtype=np.float32)
    y_var = np_to_var(100, dtype=np.float32)
    w_var = th.nn.Parameter(w_var.data)

    optim = SGD([w_var], lr=0.1)
    var_to_np(w_var * x_var)
    loss = th.abs(y_var - w_var * x_var)

    optim.zero_grad()
    loss.backward()
    # gradient will be 2 always actually
    optim.step()
    assert np.allclose(var_to_np(w_var), 3.2)


def test_adam_should_not_decay_weights_with_lr_0():
    init_w = np.float32(3)
    w_var = np_to_var(init_w, dtype=np.float64)
    x_var = np_to_var(2, dtype=np.float64)
    y_var = np_to_var(100, dtype=np.float64)
    w_var = th.nn.Parameter(w_var.data)
    lr = 0
    optim = Adam([w_var], lr=lr, weight_decay=1)

    n_epochs = 10
    for i_epoch in range(n_epochs):
        loss = th.abs(y_var - w_var * x_var)
        optim.zero_grad()
        loss.backward()
        optim.step()
        assert np.allclose(init_w, var_to_np(w_var))


def test_adamw_should_decay_weights_with_lr_0():
    init_w = np.float32(3)
    w_var = np_to_var(init_w, dtype=np.float64)
    x_var = np_to_var(2, dtype=np.float64)
    y_var = np_to_var(100, dtype=np.float64)
    w_var = th.nn.Parameter(w_var.data)
    wd = 0.1
    lr = 0
    optim = AdamW([w_var], lr=lr, weight_decay=wd)

    n_epochs = 10
    for i_epoch in range(n_epochs):
        expected_w = init_w * ((1 - wd) ** (i_epoch + 1))
        loss = th.abs(y_var - w_var * x_var)
        optim.zero_grad()
        loss.backward()
        optim.step()
        assert np.allclose(expected_w, var_to_np(w_var))


def test_adam_and_adamw_identical_without_weight_decay():
    init_w = np.float32(3)
    w_var_adam = np_to_var(init_w, dtype=np.float64)
    w_var_adamw = np_to_var(init_w, dtype=np.float64)
    x_var = np_to_var(2, dtype=np.float64)
    y_var = np_to_var(100, dtype=np.float64)
    w_var_adam = th.nn.Parameter(w_var_adam.data)
    w_var_adamw = th.nn.Parameter(w_var_adamw.data)
    lr = 0.1
    optim_adam = Adam([w_var_adam], lr=lr, weight_decay=0)
    optim_adamw = Adam([w_var_adamw], lr=lr, weight_decay=0)
    n_epochs = 10
    for i_epoch in range(n_epochs):
        loss_adam = th.abs(y_var - w_var_adam * x_var)
        optim_adam.zero_grad()
        loss_adam.backward()
        optim_adam.step()
        loss_adamw = th.abs(y_var - w_var_adamw * x_var)
        optim_adamw.zero_grad()
        loss_adamw.backward()
        optim_adamw.step()
        assert np.allclose(var_to_np(w_var_adam), var_to_np(w_var_adamw))
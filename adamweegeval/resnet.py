import numpy as np
from torch import nn
from torch.nn import init
import torch as th
from torch.nn.functional import elu
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var


class EEGResNet(object):
    """
    Residual Network for EEG.
    """
    def __init__(self, in_chans,
                 n_classes,
                 input_time_length,
                 final_pool_length,
                 n_first_filters,
                 n_layers_per_block=2,
                 first_filter_length=3,
                 nonlinearity=elu,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 batch_norm_epsilon=1e-4,
                 conv_weight_init_fn=lambda w: init.kaiming_normal(w, a=0)):
        if final_pool_length == 'auto':
            assert input_time_length is not None
        assert first_filter_length % 2 == 1
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module('dimshuffle', Expression(_transpose_time_to_spat))
            model.add_module('conv_time', nn.Conv2d(1, self.n_first_filters,
                                                    (
                                                    self.first_filter_length, 1),
                                                    stride=1,
                                                    padding=(self.first_filter_length // 2, 0)))
            model.add_module('conv_spat',
                             nn.Conv2d(self.n_first_filters, self.n_first_filters,
                                       (1, self.in_chans),
                                       stride=(1, 1),
                                       bias=False))
        else:
            model.add_module('conv_time',
                             nn.Conv2d(self.in_chans, self.n_first_filters,
                                       (self.first_filter_length, 1),
                                       stride=(1, 1),
                                       padding=(self.first_filter_length // 2, 0),
                                       bias=False,))
        n_filters_conv = self.n_first_filters
        model.add_module('bnorm',
                         nn.BatchNorm2d(n_filters_conv,
                                        momentum=self.batch_norm_alpha,
                                        affine=True,
                                        eps=1e-5),)
        model.add_module('conv_nonlin', Expression(self.nonlinearity))
        cur_dilation  = np.array([1,1])
        n_cur_filters = n_filters_conv
        i_block = 1
        for i_layer in range(self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))
        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(2* n_cur_filters)
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_out_filters,
                                           dilation=cur_dilation,))
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(1.5* n_cur_filters)
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_out_filters,
                                           dilation=cur_dilation,))
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))


        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))


        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))
        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                                 ResidualBlock(n_cur_filters, n_cur_filters,
                                               dilation=cur_dilation, ))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))



        model.eval()
        if self.final_pool_length == 'auto':
            out = model(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_pool_length = n_out_time
        model.add_module('mean_pool', AvgPool2dWithConv(
            (self.final_pool_length, 1), (1,1), dilation=(int(cur_dilation[0]),
                                                          int(cur_dilation[1]))))
        model.add_module('conv_classifier',
                             nn.Conv2d(n_cur_filters, self.n_classes,
                                       (1, 1), bias=True))
        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze',  Expression(_squeeze_final_output))


        # Initialize all weights
        model.apply(lambda module: weights_init(module, self.conv_weight_init_fn))

        # Start in eval mode
        model.eval()
        return model


def weights_init(module, conv_weight_init_fn):
    classname = module.__class__.__name__
    if 'Conv' in classname and classname != "AvgPool2dWithConv":
        conv_weight_init_fn(module.weight)
        if module.bias is not None:
            init.constant(module.bias, 0)
    elif 'BatchNorm' in classname:
        init.constant(module.weight, 1)
        init.constant(module.bias, 0)


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:,:,:,0]
    if x.size()[2] == 1:
        x = x[:,:,0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


# create a residual learning building block with two stacked 3x3 convlayers as in paper
class ResidualBlock(nn.Module):
    def __init__(
        self, in_filters,
            out_num_filters,
            dilation,
            filter_time_length=3,
            nonlinearity=elu,
            batch_norm_alpha=0.1, batch_norm_epsilon=1e-4,
        ):
        super(ResidualBlock, self).__init__()
        time_padding = int((filter_time_length - 1) * dilation[0])
        assert time_padding % 2 == 0
        time_padding = int(time_padding // 2)
        dilation = (int(dilation[0]), int(dilation[1]))
        assert (out_num_filters - in_filters) % 2 == 0, (
            "Need even number of extra channels in order to be able to "
            "pad correctly")
        self.n_pad_chans = out_num_filters - in_filters

        self.conv_1 = nn.Conv2d(
            in_filters, out_num_filters, (filter_time_length, 1), stride=(1, 1),
            dilation=dilation,
            padding=(time_padding, 0))
        self.bn1 = nn.BatchNorm2d(
            out_num_filters, momentum=batch_norm_alpha, affine=True,
            eps=batch_norm_epsilon)
        self.conv_2 = nn.Conv2d(
           out_num_filters, out_num_filters, (filter_time_length, 1), stride=(1, 1),
           dilation=dilation,
           padding=(time_padding, 0))
        self.bn2 = nn.BatchNorm2d(
           out_num_filters, momentum=batch_norm_alpha,
            affine=True, eps=batch_norm_epsilon)
        # also see https://mail.google.com/mail/u/0/#search/ilya+joos/1576137dd34c3127
        # for resnet options as ilya used them
        self.nonlinearity = nonlinearity


    def forward(self, x):
        stack_1 = self.nonlinearity(self.bn1(self.conv_1(x)))
        stack_2 = self.bn2(self.conv_2(stack_1)) # next nonlin after sum
        if self.n_pad_chans != 0:
            zeros_for_padding = th.autograd.Variable(
                th.zeros(x.size()[0], self.n_pad_chans // 2,
                         x.size()[2], x.size()[3]))
            if x.is_cuda:
                zeros_for_padding = zeros_for_padding.cuda()
            x = th.cat((zeros_for_padding, x, zeros_for_padding), dim=1)
        out = self.nonlinearity(x + stack_2)
        return out

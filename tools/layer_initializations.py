import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_layer_inverse_root_fan_in(layer):
    """ Initialize hidden layers """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def init_layer_within_range(layer, limit_range: tuple = (-3e-3, 3e-3)):
    """  We need the initialization for last layer of the Actor to be between -0.003 and 0.003 as this prevents us
     from getting 1 or -1 output values in the initial stages, which would squash our gradients to zero,
    as we use the tanh activation.
    """
    return limit_range


def get_init_layer_within_rage(limit_range: tuple = (-3e-3, 3e-3)):
    return lambda layer: init_layer_within_range(layer, limit_range=limit_range)

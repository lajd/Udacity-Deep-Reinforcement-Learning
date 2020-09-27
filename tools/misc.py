import sys
import numpy
import random
from typing import Union
import torch
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def get_object_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_object_size(v, seen) for v in obj.values()])
        size += sum([get_object_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_object_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_object_size(i, seen) for i in obj])
    return size


def concatenate_tensors(tensors_list: Union[torch.Tensor, Tensor], cat_dim=0):
    if isinstance(tensors_list, list):
        return torch.cat(tensors_list, dim=cat_dim)
    elif isinstance(tensors_list, Tensor):
        return tensors_list
    else:
        raise ValueError('Unexpected type for tensors_list: {}'.format(tensors_list))


def soft_update(online_model, target_model, tau) -> None:
    """Soft update model parameters from local to target network.

    θ_target = τ*θ_local + (1 - τ)*θ_target

    Args:
        online_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), online_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def ensure_batch(*tensor_args):
    outp = []
    for t in tensor_args:
        if t.ndim == 1:
            t = t.unsqueeze(0).to(device)
        outp.append(t)
    return outp

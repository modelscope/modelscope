# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
import importlib
import logging
import os
import warnings
from collections import OrderedDict
from inspect import signature

import torch

from modelscope.models.cv.video_depth_estimation.utils.horovod import print0
from modelscope.models.cv.video_depth_estimation.utils.misc import (make_list,
                                                                    pcolor,
                                                                    same_shape)
from modelscope.models.cv.video_depth_estimation.utils.types import is_str


def set_debug(debug):
    """
    Enable or disable debug terminal logging

    Parameters
    ----------
    debug : bool
        Debugging flag (True to enable)
    """
    # Disable logging if requested
    if not debug:
        os.environ['NCCL_DEBUG'] = ''
        os.environ['WANDB_SILENT'] = 'false'
        warnings.filterwarnings('ignore')
        logging.disable(logging.CRITICAL)


def filter_args(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function

    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering

    Returns
    -------
    filtered : dict
        Dictionary containing only keys that are arguments of func
    """
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered


def filter_args_create(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function
    and creates a function with those arguments

    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering

    Returns
    -------
    func : Function
        Function with filtered keys as arguments
    """
    return func(**filter_args(func, keys))


def load_class(filename, paths, concat=True):
    """
    Look for a file in different locations and return its method with the same name
    Optionally, you can use concat to search in path.filename instead

    Parameters
    ----------
    filename : str
        Name of the file we are searching for
    paths : str or list of str
        Folders in which the file will be searched
    concat : bool
        Flag to concatenate filename to each path during the search

    Returns
    -------
    method : Function
        Loaded method
    """
    # for each path in paths
    for path in make_list(paths):
        # Create full path
        full_path = '{}.{}'.format(path, filename) if concat else path
        if importlib.util.find_spec(full_path):
            # Return method with same name as the file
            return getattr(importlib.import_module(full_path), filename)
    raise ValueError('Unknown class {}'.format(filename))


def load_class_args_create(filename, paths, args={}, concat=True):
    """Loads a class (filename) and returns an instance with filtered arguments (args)"""
    class_type = load_class(filename, paths, concat)
    return filter_args_create(class_type, args)


def load_network(network, path, prefixes=''):
    """
    Loads a pretrained network

    Parameters
    ----------
    network : nn.Module
        Network that will receive the pretrained weights
    path : str
        File containing a 'state_dict' key with pretrained network weights
    prefixes : str or list of str
        Layer name prefixes to consider when loading the network

    Returns
    -------
    network : nn.Module
        Updated network with pretrained weights
    """
    prefixes = make_list(prefixes)
    # If path is a string
    if is_str(path):
        saved_state_dict = torch.load(path, map_location='cpu')['state_dict']
        if path.endswith('.pth.tar'):
            saved_state_dict = backwards_state_dict(saved_state_dict)
    # If state dict is already provided
    else:
        saved_state_dict = path
    # Get network state dict
    network_state_dict = network.state_dict()

    updated_state_dict = OrderedDict()
    n, n_total = 0, len(network_state_dict.keys())
    for key, val in saved_state_dict.items():
        for prefix in prefixes:
            prefix = prefix + '.'
            if prefix in key:
                idx = key.find(prefix) + len(prefix)
                key = key[idx:]
                if key in network_state_dict.keys() and \
                        same_shape(val.shape, network_state_dict[key].shape):
                    updated_state_dict[key] = val
                    n += 1
    try:
        network.load_state_dict(updated_state_dict, strict=True)
    except Exception as e:
        print(e)
        network.load_state_dict(updated_state_dict, strict=False)
    base_color, attrs = 'cyan', ['bold', 'dark']
    color = 'green' if n == n_total else 'yellow' if n > 0 else 'red'
    print0(
        pcolor(
            '=====###### Pretrained {} loaded:'.format(prefixes[0]),
            base_color,
            attrs=attrs)
        + pcolor(' {}/{} '.format(n, n_total), color, attrs=attrs)
        + pcolor('tensors', base_color, attrs=attrs))
    return network


def backwards_state_dict(state_dict):
    """
    Modify the state dict of older models for backwards compatibility

    Parameters
    ----------
    state_dict : dict
        Model state dict with pretrained weights

    Returns
    -------
    state_dict : dict
        Updated model state dict with modified layer names
    """
    # List of layer names to change
    changes = (('model.model', 'model'), ('pose_network', 'pose_net'),
               ('disp_network', 'depth_net'))
    # Iterate over all keys and values
    updated_state_dict = OrderedDict()
    for key, val in state_dict.items():
        # Ad hoc changes due to version changes
        key = '{}.{}'.format('model', key)
        if 'disp_network' in key:
            key = key.replace('conv3.0.weight', 'conv3.weight')
            key = key.replace('conv3.0.bias', 'conv3.bias')
        # Change layer names
        for change in changes:
            key = key.replace('{}.'.format(change[0]), '{}.'.format(change[1]))
        updated_state_dict[key] = val
    # Return updated state dict
    return updated_state_dict

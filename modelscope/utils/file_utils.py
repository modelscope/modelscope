# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect
import os

import cv2
import numpy as np


# TODO: remove this api, unify to flattened args
def func_receive_dict_inputs(func):
    """to decide if a func could recieve dict inputs or not

    Args:
        func (class): the target function to be inspected

    Returns:
        bool: if func only has one arg ``input`` or ``inputs``, return True, else return False
    """
    full_args_spec = inspect.getfullargspec(func)
    varargs = full_args_spec.varargs
    varkw = full_args_spec.varkw
    if not (varargs is None and varkw is None):
        return False

    args = [] if not full_args_spec.args else full_args_spec.args
    args.pop(0) if (args and args[0] in ['self', 'cls']) else args

    if len(args) == 1 and args[0] in ['input', 'inputs']:
        return True

    return False


def get_default_cache_dir():
    """
    default base dir: '~/.cache/modelscope'
    """
    default_cache_dir = os.path.expanduser(
        os.path.join('~/.cache', 'modelscope'))
    return default_cache_dir


def numpy_to_cv2img(vis_img):
    """to convert a np.array Hotmap with shape(h, w) to cv2 img

    Args:
        vis_img (np.array): input data

    Returns:
        cv2 img
    """
    vis_img = (vis_img - vis_img.min()) / (
        vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    return vis_img

import json
import os.path as osp
import time
import torch
import numpy as np

# private package
from ..conf import *
from .backbone import StackedHGNetV1


def get_config(args):
    config = None
    config_name = args.config_name
    if config_name == "alignment":
        config = Alignment(args)
    else:
        assert NotImplementedError

    return config


def get_net(config):
    net = None
    if config.net == "stackedHGnet_v1":
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)
    else:
        assert False
    return net


def set_environment(config):
    if config.device_id >= 0:
        assert torch.cuda.is_available() and torch.cuda.device_count() > config.device_id
        torch.cuda.empty_cache()
        config.device = torch.device("cuda", config.device_id)
        config.use_gpu = True
    else:
        config.device = torch.device("cpu")
        config.use_gpu = False

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_flush_denormal(True)  # ignore extremely small value
    torch.backends.cudnn.benchmark = True  # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.autograd.set_detect_anomaly(True)


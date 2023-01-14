# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
import importlib
import random
from collections import OrderedDict

import numpy as np
import torch

from modelscope.models.cv.video_depth_estimation.utils.load import (
    filter_args, load_class, load_class_args_create, load_network)
from modelscope.models.cv.video_depth_estimation.utils.misc import pcolor


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, logger=None, load_datasets=True):
        super().__init__()

        # Store configuration, checkpoint and logger
        self.config = config
        self.logger = logger
        self.resume = resume

        # Set random seed
        set_random_seed(config.arch.seed)

        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1',
                             'a2', 'a3', 'SILog', 'l1_inv', 'rot_ang', 't_ang',
                             't_cm')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        # Model, optimizers, schedulers and datasets are None for now
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0

        # Prepare model
        self.prepare_model(resume)

        # Preparations done
        self.config.prepared = True

    def prepare_model(self, resume=None):
        """Prepare self.model (incl. loading previous state)"""
        print0(pcolor('### Preparing Model', 'green'))
        self.model = setup_model(self.config.model, self.config.prepared)
        # Resume model if available
        if resume:
            print0(
                pcolor(
                    '### Resuming from {}'.format(resume['file']),
                    'magenta',
                    attrs=['bold']))
            self.model = load_network(self.model, resume['state_dict'],
                                      'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch']

    @property
    def depth_net(self):
        """Returns depth network."""
        return self.model.depth_net

    @property
    def pose_net(self):
        """Returns pose network."""
        return self.model.pose_net

    @property
    def percep_net(self):
        """Returns perceptual network."""
        return self.model.percep_net

    @property
    def logs(self):
        """Returns various logs for tracking."""
        params = OrderedDict()
        for param in self.optimizer.param_groups:
            params['{}_learning_rate'.format(
                param['name'].lower())] = param['lr']
        params['progress'] = self.progress
        return {
            **params,
            **self.model.logs,
        }

    @property
    def progress(self):
        """Returns training progress (current epoch / max. number of epochs)"""
        return self.current_epoch / self.config.arch.max_epochs

    def configure_optimizers(self):
        """Configure depth and pose optimizers and the corresponding scheduler."""

        params = []
        # Load optimizer
        optimizer = getattr(torch.optim, self.config.model.optimizer.name)
        # Depth optimizer
        if self.depth_net is not None:
            params.append({
                'name':
                'Depth',
                'params':
                self.depth_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.depth)
            })
        # Pose optimizer
        if self.pose_net is not None:
            params.append({
                'name':
                'Pose',
                'params': [
                    param for param in self.pose_net.parameters()
                    if param.requires_grad
                ],
                **filter_args(optimizer, self.config.model.optimizer.pose)
            })
        # Create optimizer with parameters
        optimizer = optimizer(params)

        # Load and initialize scheduler
        scheduler = getattr(torch.optim.lr_scheduler,
                            self.config.model.scheduler.name)
        scheduler = scheduler(
            optimizer, **filter_args(scheduler, self.config.model.scheduler))

        # Create class variables so we can use it internally
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Return optimizer and scheduler
        return optimizer, scheduler

    def forward(self, *args, **kwargs):
        """Runs the model and returns the output."""
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        """Runs the pose network and returns the output."""
        assert self.depth_net is not None, 'Depth network not defined'
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.pose_net is not None, 'Pose network not defined'
        return self.pose_net(*args, **kwargs)

    def percep(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.percep_net is not None, 'Perceptual network not defined'
        return self.percep_net(*args, **kwargs)


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_depth_net(config, prepared, **kwargs):
    """
    Create a depth network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    print0(pcolor('DepthNet: %s' % config.name, 'yellow'))
    if config.name == 'DepthPoseNet':
        model_class = getattr(
            importlib.import_module(
                'modelscope.models.cv.video_depth_estimation.networks.depth_pose.depth_pose_net'
            ), 'DepthPoseNet')
    depth_net = model_class(**{**config, **kwargs})
    if not prepared and config.checkpoint_path != '':
        depth_net = load_network(depth_net, config.checkpoint_path,
                                 ['depth_net', 'disp_network'])
    return depth_net


def setup_pose_net(config, prepared, **kwargs):
    """
    Create a pose network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    pose_net : nn.Module
        Created pose network
    """
    print0(pcolor('PoseNet: %s' % config.name, 'yellow'))
    pose_net = load_class_args_create(
        config.name,
        paths=[
            'modelscope.models.cv.video_depth_estimation.networks.pose',
        ],
        args={
            **config,
            **kwargs
        },
    )
    if not prepared and config.checkpoint_path != '':
        pose_net = load_network(pose_net, config.checkpoint_path,
                                ['pose_net', 'pose_network'])
    return pose_net


def setup_percep_net(config, prepared, **kwargs):
    """
    Create a perceputal network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    print0(pcolor('PercepNet: %s' % config.name, 'yellow'))
    percep_net = load_class_args_create(
        config.name,
        paths=[
            'modelscope.models.cv.video_depth_estimation.networks.layers',
        ],
        args={
            **config,
            **kwargs
        },
    )
    return percep_net


def setup_model(config, prepared, **kwargs):
    """
    Create a model

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    prepared : bool
        True if the model has been prepared before
    kwargs : dict
        Extra parameters for the model

    Returns
    -------
    model : nn.Module
        Created model
    """
    print0(pcolor('Model: %s' % config.name, 'yellow'))
    config.loss.min_depth = config.params.min_depth
    config.loss.max_depth = config.params.max_depth
    if config.name == 'SupModelMF':
        model_class = getattr(
            importlib.import_module(
                'modelscope.models.cv.video_depth_estimation.models.sup_model_mf'
            ), 'SupModelMF')
    model = model_class(**{**config.loss, **kwargs})
    # Add depth network if required
    if model.network_requirements['depth_net']:
        config.depth_net.max_depth = config.params.max_depth
        config.depth_net.min_depth = config.params.min_depth
        model.add_depth_net(setup_depth_net(config.depth_net, prepared))
    # Add pose network if required
    if model.network_requirements['pose_net']:
        model.add_pose_net(setup_pose_net(config.pose_net, prepared))
    # Add percep_net if required
    if model.network_requirements['percep_net']:
        model.add_percep_net(setup_percep_net(config.percep_net, prepared))
    # If a checkpoint is provided, load pretrained model
    if not prepared and config.checkpoint_path != '':
        model = load_network(model, config.checkpoint_path, 'model')
    # Return model
    return model


def print0(string='\n'):
    print(string)

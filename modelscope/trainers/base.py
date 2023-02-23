# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from modelscope.hub.check_model import check_local_model_is_latest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.config import Config
from modelscope.utils.constant import Invoke, ThirdParty
from .utils.log_buffer import LogBuffer


class BaseTrainer(ABC):
    """ Base class for trainer which can not be instantiated.

    BaseTrainer defines necessary interface
    and provide default implementation for basic initialization
    such as parsing config file and parsing commandline args.
    """

    def __init__(self, cfg_file: str, arg_parse_fn: Optional[Callable] = None):
        """ Trainer basic init, should be called in derived class

        Args:
            cfg_file: Path to configuration file.
            arg_parse_fn: Same as ``parse_fn`` in :obj:`Config.to_args`.
        """
        self.cfg = Config.from_file(cfg_file)
        if arg_parse_fn:
            self.args = self.cfg.to_args(arg_parse_fn)
        else:
            self.args = None
        self.log_buffer = LogBuffer()
        self.visualization_buffer = LogBuffer()
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    def get_or_download_model_dir(self,
                                  model,
                                  model_revision=None,
                                  third_party=None):
        """ Get local model directory or download model if necessary.

        Args:
            model (str): model id or path to local model directory.
            model_revision  (str, optional): model version number.
            third_party (str, optional): in which third party library
                this function is called.
        """
        if os.path.exists(model):
            model_cache_dir = model if os.path.isdir(
                model) else os.path.dirname(model)
            check_local_model_is_latest(
                model_cache_dir,
                user_agent={
                    Invoke.KEY: Invoke.LOCAL_TRAINER,
                    ThirdParty.KEY: third_party
                })
        else:
            model_cache_dir = snapshot_download(
                model,
                revision=model_revision,
                user_agent={
                    Invoke.KEY: Invoke.TRAINER,
                    ThirdParty.KEY: third_party
                })
        return model_cache_dir

    @abstractmethod
    def train(self, *args, **kwargs):
        """ Train (and evaluate) process

        Train process should be implemented for specific task or
        model, related parameters have been initialized in
        ``BaseTrainer.__init__`` and should be used in this function
        """
        pass

    @abstractmethod
    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        """ Evaluation process

        Evaluation process should be implemented for specific task or
        model, related parameters have been initialized in
        ``BaseTrainer.__init__`` and should be used in this function
        """
        pass


@TRAINERS.register_module(module_name='dummy')
class DummyTrainer(BaseTrainer):

    def __init__(self, cfg_file: str, *args, **kwargs):
        """ Dummy Trainer.

        Args:
            cfg_file: Path to configuration file.
        """
        super().__init__(cfg_file)

    def train(self, *args, **kwargs):
        """ Train (and evaluate) process

        Train process should be implemented for specific task or
        model, related parameters have been initialized in
        ``BaseTrainer.__init__`` and should be used in this function
        """
        cfg = self.cfg.train
        print(f'train cfg {cfg}')

    def evaluate(self,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        """ Evaluation process

        Evaluation process should be implemented for specific task or
        model, related parameters have been initialized in
        ``BaseTrainer.__init__`` and should be used in this function
        """
        cfg = self.cfg.evaluation
        print(f'eval cfg {cfg}')
        print(f'checkpoint_path {checkpoint_path}')

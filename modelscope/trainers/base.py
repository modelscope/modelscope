# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from modelscope.trainers.builder import TRAINERS
from modelscope.utils.config import Config
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
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    @abstractmethod
    def train(self, *args, **kwargs):
        """ Train (and evaluate) process

        Train process should be implemented for specific task or
        model, releated paramters have been intialized in
        ``BaseTrainer.__init__`` and should be used in this function
        """
        pass

    @abstractmethod
    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        """ Evaluation process

        Evaluation process should be implemented for specific task or
        model, releated paramters have been intialized in
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
        model, releated paramters have been intialized in
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
        model, releated paramters have been intialized in
        ``BaseTrainer.__init__`` and should be used in this function
        """
        cfg = self.cfg.evaluation
        print(f'eval cfg {cfg}')
        print(f'checkpoint_path {checkpoint_path}')

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.hub import read_config
from modelscope.utils.logger import get_logger
from .bnext import BNext

logger = get_logger()

__all__ = ['BinaryQuantClassificationModel']


@MODELS.register_module(Tasks.image_classification, module_name=Models.bnext)
class BinaryQuantClassificationModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            logger.info('Use GPU: {}'.format(self._device))
        else:
            self._device = torch.device('cpu')
            logger.info('Use CPU: {}'.format(self._device))

        self.model = BNext(num_classes=1000)
        self.model = self.model.to(self._device)

        self.model_dir = model_dir

        self._load_pretrained_checkpoint()

    def forward(self, inputs, return_loss=False):

        return self.model(**inputs)

    def _convert_state_dict(self, state_dict):
        """Converts a state dict saved from a dataParallel module to normal
        module state_dict inplace
        :param state_dict is the loaded DataParallel model_state
        """
        if not next(iter(state_dict)).startswith('module.'):
            return state_dict  # abort if dict is not a DataParallel model_state
        new_state_dict = OrderedDict()

        split_index = 0
        for cur_key, _ in state_dict.items():
            if cur_key.startswith('module.model'):
                split_index = 13
            elif cur_key.startswith('module'):
                split_index = 7
            break

        for k, v in state_dict.items():
            name = k[split_index:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def _load_pretrained_checkpoint(self):
        model_path = os.path.join(self.model_dir, ModelFile.TORCH_MODEL_FILE)
        logger.info(model_path)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, 'cpu')
            model_state = self._convert_state_dict(ckpt['state_dict'])

            if ckpt.get('meta', None):
                self.CLASSES = ckpt['meta']
                self.config_type = 'ms_config'
            self.model.load_state_dict(model_state)
            self.model.to(self._device)

        else:
            logger.error(
                '[checkModelPath]:model path dose not exits!!! model Path:'
                + model_path)
            raise Exception('[checkModelPath]:model path dose not exits!')

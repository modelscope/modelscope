# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaTextToImageSynthesisPreprocessor(OfaBasePreprocessor):

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaTextToImageSynthesisPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)
        self.max_src_length = 64

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source = ' '.join(
            data['text'].lower().strip().split()[:self.max_src_length])
        source = 'what is the complete image? caption: {}'.format(source)
        inputs = self.get_inputs(source)
        sample = {
            'source': inputs,
            'patch_images': None,
            'patch_masks': torch.tensor([False]),
            'code_masks': torch.tensor([False])
        }
        return sample

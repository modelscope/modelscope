# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaTextToImageSynthesisPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for text to image synthesis tasks.
    """

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
        r"""
        Building samples for inference.

        step 1. Preprocessing for str input.
            - do lower, strip and restrict the total length by `max_src_length`.
        step 2. Building text to image synthesis instruction. The template of
            the instruction is like `what is the complete image? caption: {}`,
            while the `{}` will be replaced by the result of step 1.
        step 3. Tokenize the instruction as model's inputs.


        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `text`,
                which refer to the description of synthesis image.
        Return:
            A dict object, contains source text input, patch images with `None` value
            patch masks and code masks with `Tensor([False])` value.
        """
        source = ' '.join(
            data['text'].lower().strip().split()[:self.max_src_length])
        source = 'what is the complete image? caption: {}'.format(source)
        inputs = self.tokenize_text(source)
        sample = {
            'source': inputs,
            'patch_images': None,
            'patch_masks': torch.tensor([False]),
            'code_masks': torch.tensor([False])
        }
        return sample

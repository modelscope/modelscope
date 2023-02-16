# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Dict, Union

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .ddcolor import DDColor

logger = get_logger()

__all__ = ['DDColorForImageColorization']


@MODELS.register_module(Tasks.image_colorization, module_name=Models.ddcolor)
class DDColorForImageColorization(TorchModel):

    def __init__(self,
                 model_dir,
                 encoder_name='convnext-l',
                 input_size=(512, 512),
                 num_queries=100,
                 *args,
                 **kwargs):
        """initialize the image colorization model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            encoder_name (str): the encoder name.
            input_size (tuple): size of the model input image.
            num_queries (int): number of decoder queries
        """
        super().__init__(model_dir, *args, **kwargs)

        self.model = DDColor(encoder_name, input_size, num_queries)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = self._load_pretrained(self.model, model_path)

    def forward(self, input: Dict[str,
                                  Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result of the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """
        return self.model(**input)

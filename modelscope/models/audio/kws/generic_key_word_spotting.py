import os
from typing import Any, Dict

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

__all__ = ['GenericKeyWordSpotting']


@MODELS.register_module(Tasks.key_word_spotting, module_name=Models.kws_kwsbp)
class GenericKeyWordSpotting(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the info of model.

        Args:
            model_dir (str): the model path.
        """

        self.model_cfg = {
            'model_workspace': model_dir,
            'config_path': os.path.join(model_dir, 'config.yaml')
        }

    def forward(self) -> Dict[str, Any]:
        """return the info of the model
        """
        return self.model_cfg

import os
from typing import Any, Dict

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

__all__ = ['GenericAutomaticSpeechRecognition']


@MODELS.register_module(
    Tasks.auto_speech_recognition, module_name=Models.generic_asr)
class GenericAutomaticSpeechRecognition(Model):

    def __init__(self, model_dir: str, am_model_name: str,
                 model_config: Dict[str, Any], *args, **kwargs):
        """initialize the info of model.

        Args:
            model_dir (str): the model path.
            am_model_name (str): the am model name from configuration.json
        """

        self.model_cfg = {
            # the recognition model dir path
            'model_workspace': model_dir,
            # the am model name
            'am_model': am_model_name,
            # the am model file path
            'am_model_path': os.path.join(model_dir, am_model_name),
            # the recognition model config dict
            'model_config': model_config
        }

    def forward(self) -> Dict[str, Any]:
        """return the info of the model
        """
        return self.model_cfg

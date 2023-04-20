# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tempfile
from typing import Dict, Optional

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.audio.audio_utils import update_conf
from modelscope.utils.constant import Tasks
from .fsmn_sele_v2 import FSMNSeleNetV2
from .fsmn_sele_v3 import FSMNSeleNetV3


@MODELS.register_module(
    Tasks.keyword_spotting, module_name=Models.speech_dfsmn_kws_char_farfield)
class FSMNSeleNetV2Decorator(TorchModel):
    r""" A decorator of FSMNSeleNetV2 for integrating into modelscope framework """

    MODEL_CLASS = FSMNSeleNetV2
    MODEL_TXT = 'model.txt'
    SC_CONFIG = 'sound_connect.conf'

    def __init__(self,
                 model_dir: str,
                 training: Optional[bool] = False,
                 *args,
                 **kwargs):
        """initialize the dfsmn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        if training:
            self.model = self.MODEL_CLASS(*args, **kwargs)
        else:
            sc_config_file = os.path.join(model_dir, self.SC_CONFIG)
            model_txt_file = os.path.join(model_dir, self.MODEL_TXT)
            self.tmp_dir = tempfile.TemporaryDirectory()
            new_config_file = os.path.join(self.tmp_dir.name, self.SC_CONFIG)

            self._sc = None
            if os.path.exists(model_txt_file):
                conf_dict = dict(kws_model=model_txt_file)
                update_conf(sc_config_file, new_config_file, conf_dict)
                import py_sound_connect
                self._sc = py_sound_connect.SoundConnect(new_config_file)
                self.size_in = self._sc.bytesPerBlockIn()
                self.size_out = self._sc.bytesPerBlockOut()
            else:
                raise Exception(
                    f'Invalid model directory! Failed to load model file:'
                    f' {model_txt_file}.')

    def __del__(self):
        if hasattr(self, 'tmp_dir'):
            self.tmp_dir.cleanup()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model.forward(input)

    def forward_decode(self, data: bytes):
        result = {'pcm': self._sc.process(data, self.size_out)}
        state = self._sc.kwsState()
        if state == 2:
            result['kws'] = {
                'keyword':
                self._sc.kwsKeyword(self._sc.kwsSpottedKeywordIndex()),
                'offset': self._sc.kwsKeywordOffset(),
                'channel': self._sc.kwsBestChannel(),
                'length': self._sc.kwsKeywordLength(),
                'confidence': self._sc.kwsConfidence()
            }
        return result


@MODELS.register_module(
    Tasks.keyword_spotting,
    module_name=Models.speech_dfsmn_kws_char_farfield_iot)
class FSMNSeleNetV3Decorator(FSMNSeleNetV2Decorator):
    r""" A decorator of FSMNSeleNetV3 for integrating into modelscope framework """

    MODEL_CLASS = FSMNSeleNetV3

    def __init__(self,
                 model_dir: str,
                 training: Optional[bool] = False,
                 *args,
                 **kwargs):
        """initialize the dfsmn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, training, *args, **kwargs)

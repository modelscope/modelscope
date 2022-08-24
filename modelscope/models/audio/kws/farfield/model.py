import os
from typing import Dict

import torch

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .fsmn_sele_v2 import FSMNSeleNetV2


@MODELS.register_module(
    Tasks.keyword_spotting, module_name=Models.speech_dfsmn_kws_char_farfield)
class FSMNSeleNetV2Decorator(TorchModel):
    r""" A decorator of FSMNSeleNetV2 for integrating into modelscope framework """

    MODEL_TXT = 'model.txt'
    SC_CONFIG = 'sound_connect.conf'
    SC_CONF_ITEM_KWS_MODEL = '${kws_model}'

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the dfsmn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        sc_config_file = os.path.join(model_dir, self.SC_CONFIG)
        model_txt_file = os.path.join(model_dir, self.MODEL_TXT)
        model_bin_file = os.path.join(model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE)
        self._model = None
        if os.path.exists(model_bin_file):
            kwargs.pop('device')
            self._model = FSMNSeleNetV2(*args, **kwargs)
            checkpoint = torch.load(model_bin_file)
            self._model.load_state_dict(checkpoint, strict=False)

        self._sc = None
        if os.path.exists(model_txt_file):
            with open(sc_config_file) as f:
                lines = f.readlines()
            with open(sc_config_file, 'w') as f:
                for line in lines:
                    if self.SC_CONF_ITEM_KWS_MODEL in line:
                        line = line.replace(self.SC_CONF_ITEM_KWS_MODEL,
                                            model_txt_file)
                    f.write(line)
            import py_sound_connect
            self._sc = py_sound_connect.SoundConnect(sc_config_file)
            self.size_in = self._sc.bytesPerBlockIn()
            self.size_out = self._sc.bytesPerBlockOut()

        if self._model is None and self._sc is None:
            raise Exception(
                f'Invalid model directory! Neither {model_txt_file} nor {model_bin_file} exists.'
            )

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ...

    def forward_decode(self, data: bytes):
        result = {'pcm': self._sc.process(data, self.size_out)}
        state = self._sc.kwsState()
        if state == 2:
            result['kws'] = {
                'keyword':
                self._sc.kwsKeyword(self._sc.kwsSpottedKeywordIndex()),
                'offset': self._sc.kwsKeywordOffset(),
                'length': self._sc.kwsKeywordLength(),
                'confidence': self._sc.kwsConfidence()
            }
        return result

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import glob
import os
import time

import json
import numpy as np
import torch
from scipy.io.wavfile import write

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.audio.tts_exceptions import \
    TtsVocoderMelspecShapeMismatchException
from modelscope.utils.constant import ModelFile, Tasks
from .models import Generator

__all__ = ['Hifigan16k', 'AttrDict']
MAX_WAV_VALUE = 32768.0


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print('Complete.')
    return checkpoint_dict


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@MODELS.register_module(Tasks.text_to_speech, module_name=Models.hifigan16k)
class Hifigan16k(Model):

    def __init__(self, model_dir, *args, **kwargs):
        self._ckpt_path = os.path.join(model_dir,
                                       ModelFile.TORCH_MODEL_BIN_FILE)
        self._config = AttrDict(**kwargs)

        super().__init__(self._ckpt_path, *args, **kwargs)
        if torch.cuda.is_available():
            torch.manual_seed(self._config.seed)
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._generator = Generator(self._config).to(self._device)
        state_dict_g = load_checkpoint(self._ckpt_path, self._device)
        self._generator.load_state_dict(state_dict_g['generator'])
        self._generator.eval()
        self._generator.remove_weight_norm()

    def forward(self, melspec):
        dim0 = list(melspec.shape)[-1]
        if dim0 != 80:
            raise TtsVocoderMelspecShapeMismatchException(
                'input melspec mismatch 0 dim require 80 but {}'.format(dim0))
        with torch.no_grad():
            x = melspec.T
            x = torch.FloatTensor(x).to(self._device)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            y_g_hat = self._generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            return audio

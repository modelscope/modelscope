# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import pickle as pkl

import json
import numpy as np
import torch

from modelscope.utils.audio.tts_exceptions import \
    TtsModelConfigurationException
from modelscope.utils.constant import ModelFile, Tasks
from .models.datasets.units import KanTtsLinguisticUnit
from .models.models.hifigan import Generator
from .models.models.sambert import KanTtsSAMBERT
from .models.utils import (AttrDict, build_env, init_weights, load_checkpoint,
                           plot_spectrogram, save_checkpoint, scan_checkpoint)

MAX_WAV_VALUE = 32768.0


class Voice:

    def __init__(self, voice_name, voice_path, am_config, voc_config):
        self.__voice_name = voice_name
        self.__voice_path = voice_path
        self.__am_config = AttrDict(**am_config)
        self.__voc_config = AttrDict(**voc_config)
        self.__model_loaded = False
        if 'am' not in self.__am_config:
            raise TtsModelConfigurationException(
                'modelscope error: am configuration invalid')
        if 'linguistic_unit' not in self.__am_config:
            raise TtsModelConfigurationException(
                'modelscope error: am configuration invalid')
        self.__am_lingustic_unit_config = self.__am_config['linguistic_unit']

    def __load_am(self):
        local_am_ckpt_path = os.path.join(self.__voice_path, 'am')
        self.__am_ckpt_path = os.path.join(local_am_ckpt_path,
                                           ModelFile.TORCH_MODEL_BIN_FILE)
        has_mask = True
        if 'has_mask' in self.__am_lingustic_unit_config:
            has_mask = self.__am_lingustic_unit_config.has_mask
        self.__ling_unit = KanTtsLinguisticUnit(
            self.__am_lingustic_unit_config, self.__voice_path, has_mask)
        self.__am_net = KanTtsSAMBERT(self.__am_config,
                                      self.__ling_unit.get_unit_size()).to(
                                          self.__device)
        state_dict_g = {}
        try:
            state_dict_g = load_checkpoint(self.__am_ckpt_path, self.__device)
        except RuntimeError:
            with open(self.__am_ckpt_path, 'rb') as f:
                pth_var_dict = pkl.load(f)
                state_dict_g['fsnet'] = {
                    k: torch.FloatTensor(v)
                    for k, v in pth_var_dict['fsnet'].items()
                }
        self.__am_net.load_state_dict(state_dict_g['fsnet'], strict=False)
        self.__am_net.eval()

    def __load_vocoder(self):
        local_voc_ckpy_path = os.path.join(self.__voice_path, 'vocoder')
        self.__voc_ckpt_path = os.path.join(local_voc_ckpy_path,
                                            ModelFile.TORCH_MODEL_BIN_FILE)
        self.__generator = Generator(self.__voc_config).to(self.__device)
        state_dict_g = load_checkpoint(self.__voc_ckpt_path, self.__device)
        self.__generator.load_state_dict(state_dict_g['generator'])
        self.__generator.eval()
        self.__generator.remove_weight_norm()

    def __am_forward(self, symbol_seq):
        with torch.no_grad():
            inputs_feat_lst = self.__ling_unit.encode_symbol_sequence(
                symbol_seq)
            inputs_sy = torch.from_numpy(inputs_feat_lst[0]).long().to(
                self.__device)
            inputs_tone = torch.from_numpy(inputs_feat_lst[1]).long().to(
                self.__device)
            inputs_syllable = torch.from_numpy(inputs_feat_lst[2]).long().to(
                self.__device)
            inputs_ws = torch.from_numpy(inputs_feat_lst[3]).long().to(
                self.__device)
            inputs_ling = torch.stack(
                [inputs_sy, inputs_tone, inputs_syllable, inputs_ws],
                dim=-1).unsqueeze(0)
            inputs_emo = torch.from_numpy(inputs_feat_lst[4]).long().to(
                self.__device).unsqueeze(0)
            inputs_spk = torch.from_numpy(inputs_feat_lst[5]).long().to(
                self.__device).unsqueeze(0)
            inputs_len = torch.zeros(1).to(self.__device).long(
            ) + inputs_emo.size(1) - 1  # minus 1 for "~"
            res = self.__am_net(inputs_ling[:, :-1, :], inputs_emo[:, :-1],
                                inputs_spk[:, :-1], inputs_len)
            postnet_outputs = res['postnet_outputs']
            LR_length_rounded = res['LR_length_rounded']
            valid_length = int(LR_length_rounded[0].item())
            postnet_outputs = postnet_outputs[
                0, :valid_length, :].cpu().numpy()
            return postnet_outputs

    def __vocoder_forward(self, melspec):
        dim0 = list(melspec.shape)[-1]
        if dim0 != self.__voc_config.num_mels:
            raise TtsVocoderMelspecShapeMismatchException(
                'modelscope error: input melspec mismatch require {} but {}'.
                format(self.__voc_config.num_mels, dim0))
        with torch.no_grad():
            x = melspec.T
            x = torch.FloatTensor(x).to(self.__device)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            y_g_hat = self.__generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            return audio

    def forward(self, symbol_seq):
        if not self.__model_loaded:
            torch.manual_seed(self.__am_config.seed)
            if torch.cuda.is_available():
                torch.manual_seed(self.__am_config.seed)
                self.__device = torch.device('cuda')
            else:
                self.__device = torch.device('cpu')
            self.__load_am()
            self.__load_vocoder()
            self.__model_loaded = True
        return self.__vocoder_forward(self.__am_forward(symbol_seq))

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import io
import os
import time
import zipfile
from typing import Any, Dict, Optional, Union

import json
import numpy as np
import tensorflow as tf
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.audio.tts_exceptions import (
    TtsFrontendInitializeFailedException,
    TtsFrontendLanguageTypeInvalidException, TtsModelConfigurationExcetion,
    TtsVocoderMelspecShapeMismatchException)
from modelscope.utils.constant import ModelFile, Tasks
from .models import Generator, create_am_model
from .text.symbols import load_symbols
from .text.symbols_dict import SymbolsDict

__all__ = ['SambertHifigan']
MAX_WAV_VALUE = 32768.0


def multi_label_symbol_to_sequence(my_classes, my_symbol):
    one_hot = MultiLabelBinarizer(classes=my_classes)
    tokens = my_symbol.strip().split(' ')
    sequences = []
    for token in tokens:
        sequences.append(tuple(token.split('&')))
    return one_hot.fit_transform(sequences)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@MODELS.register_module(
    Tasks.text_to_speech, module_name=Models.sambert_hifigan)
class SambertHifigan(Model):

    def __init__(self,
                 model_dir,
                 pitch_control_str='',
                 duration_control_str='',
                 energy_control_str='',
                 *args,
                 **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        if 'am' not in kwargs:
            raise TtsModelConfigurationExcetion(
                'configuration model field missing am!')
        if 'vocoder' not in kwargs:
            raise TtsModelConfigurationExcetion(
                'configuration model field missing vocoder!')
        if 'lang_type' not in kwargs:
            raise TtsModelConfigurationExcetion(
                'configuration model field missing lang_type!')
        # initialize frontend
        import ttsfrd
        frontend = ttsfrd.TtsFrontendEngine()
        zip_file = os.path.join(model_dir, 'resource.zip')
        self._res_path = os.path.join(model_dir, 'resource')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        if not frontend.initialize(self._res_path):
            raise TtsFrontendInitializeFailedException(
                'resource invalid: {}'.format(self._res_path))
        if not frontend.set_lang_type(kwargs['lang_type']):
            raise TtsFrontendLanguageTypeInvalidException(
                'language type invalid: {}'.format(kwargs['lang_type']))
        self._frontend = frontend

        # initialize am
        tf.reset_default_graph()
        local_am_ckpt_path = os.path.join(ModelFile.TF_CHECKPOINT_FOLDER,
                                          'ckpt')
        self._am_ckpt_path = os.path.join(model_dir, local_am_ckpt_path)
        self._dict_path = os.path.join(model_dir, 'dicts')
        self._am_hparams = tf.contrib.training.HParams(**kwargs['am'])
        has_mask = True
        if self._am_hparams.get('has_mask') is not None:
            has_mask = self._am_hparams.has_mask
            print('set has_mask to {}'.format(has_mask))
        values = self._am_hparams.values()
        hp = [' {}:{}'.format(name, values[name]) for name in sorted(values)]
        print('Hyperparameters:\n' + '\n'.join(hp))
        model_name = 'robutrans'
        self._lfeat_type_list = self._am_hparams.lfeat_type_list.strip().split(
            ',')
        sy, tone, syllable_flag, word_segment, emo_category, speaker = load_symbols(
            self._dict_path, has_mask)
        self._sy = sy
        self._tone = tone
        self._syllable_flag = syllable_flag
        self._word_segment = word_segment
        self._emo_category = emo_category
        self._speaker = speaker
        self._inputs_dim = dict()
        for lfeat_type in self._lfeat_type_list:
            if lfeat_type == 'sy':
                self._inputs_dim[lfeat_type] = len(sy)
            elif lfeat_type == 'tone':
                self._inputs_dim[lfeat_type] = len(tone)
            elif lfeat_type == 'syllable_flag':
                self._inputs_dim[lfeat_type] = len(syllable_flag)
            elif lfeat_type == 'word_segment':
                self._inputs_dim[lfeat_type] = len(word_segment)
            elif lfeat_type == 'emo_category':
                self._inputs_dim[lfeat_type] = len(emo_category)
            elif lfeat_type == 'speaker':
                self._inputs_dim[lfeat_type] = len(speaker)

        self._symbols_dict = SymbolsDict(sy, tone, syllable_flag, word_segment,
                                         emo_category, speaker,
                                         self._inputs_dim,
                                         self._lfeat_type_list)
        dim_inputs = sum(self._inputs_dim.values(
        )) - self._inputs_dim['speaker'] - self._inputs_dim['emo_category']
        inputs = tf.placeholder(tf.float32, [1, None, dim_inputs], 'inputs')
        inputs_emotion = tf.placeholder(
            tf.float32, [1, None, self._inputs_dim['emo_category']],
            'inputs_emotion')
        inputs_speaker = tf.placeholder(tf.float32,
                                        [1, None, self._inputs_dim['speaker']],
                                        'inputs_speaker')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        pitch_contours_scale = tf.placeholder(tf.float32, [1, None],
                                              'pitch_contours_scale')
        energy_contours_scale = tf.placeholder(tf.float32, [1, None],
                                               'energy_contours_scale')
        duration_scale = tf.placeholder(tf.float32, [1, None],
                                        'duration_scale')
        with tf.variable_scope('model') as _:
            self._model = create_am_model(model_name, self._am_hparams)
            self._model.initialize(
                inputs,
                inputs_emotion,
                inputs_speaker,
                input_lengths,
                duration_scales=duration_scale,
                pitch_scales=pitch_contours_scale,
                energy_scales=energy_contours_scale)
            self._mel_spec = self._model.mel_outputs[0]
            self._duration_outputs = self._model.duration_outputs[0]
            self._duration_outputs_ = self._model.duration_outputs_[0]
            self._pitch_contour_outputs = self._model.pitch_contour_outputs[0]
            self._energy_contour_outputs = self._model.energy_contour_outputs[
                0]
            self._embedded_inputs_emotion = self._model.embedded_inputs_emotion[
                0]
            self._embedding_fsmn_outputs = self._model.embedding_fsmn_outputs[
                0]
            self._encoder_outputs = self._model.encoder_outputs[0]
            self._pitch_embeddings = self._model.pitch_embeddings[0]
            self._energy_embeddings = self._model.energy_embeddings[0]
            self._LR_outputs = self._model.LR_outputs[0]
            self._postnet_fsmn_outputs = self._model.postnet_fsmn_outputs[0]
            self._attention_h = self._model.attention_h
            self._attention_x = self._model.attention_x

            print('Loading checkpoint: %s' % self._am_ckpt_path)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config=config)
            self._session.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(self._session, self._am_ckpt_path)

            duration_cfg_lst = []
            if len(duration_control_str) != 0:
                for item in duration_control_str.strip().split('|'):
                    percent, scale = item.lstrip('(').rstrip(')').split(',')
                    duration_cfg_lst.append((float(percent), float(scale)))

            self._duration_cfg_lst = duration_cfg_lst

            pitch_contours_cfg_lst = []
            if len(pitch_control_str) != 0:
                for item in pitch_control_str.strip().split('|'):
                    percent, scale = item.lstrip('(').rstrip(')').split(',')
                    pitch_contours_cfg_lst.append(
                        (float(percent), float(scale)))

            self._pitch_contours_cfg_lst = pitch_contours_cfg_lst

            energy_contours_cfg_lst = []
            if len(energy_control_str) != 0:
                for item in energy_control_str.strip().split('|'):
                    percent, scale = item.lstrip('(').rstrip(')').split(',')
                    energy_contours_cfg_lst.append(
                        (float(percent), float(scale)))

            self._energy_contours_cfg_lst = energy_contours_cfg_lst

        # initialize vocoder
        self._voc_ckpt_path = os.path.join(model_dir,
                                           ModelFile.TORCH_MODEL_BIN_FILE)
        self._voc_config = AttrDict(**kwargs['vocoder'])
        print(self._voc_config)
        if torch.cuda.is_available():
            torch.manual_seed(self._voc_config.seed)
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._generator = Generator(self._voc_config).to(self._device)
        state_dict_g = load_checkpoint(self._voc_ckpt_path, self._device)
        self._generator.load_state_dict(state_dict_g['generator'])
        self._generator.eval()
        self._generator.remove_weight_norm()

    def am_synthesis_one_sentences(self, text):
        cleaner_names = [
            x.strip() for x in self._am_hparams.cleaners.split(',')
        ]

        lfeat_symbol = text.strip().split(' ')
        lfeat_symbol_separate = [''] * int(len(self._lfeat_type_list))
        for this_lfeat_symbol in lfeat_symbol:
            this_lfeat_symbol = this_lfeat_symbol.strip('{').strip('}').split(
                '$')
            if len(this_lfeat_symbol) != len(self._lfeat_type_list):
                raise Exception(
                    'Length of this_lfeat_symbol in training data'
                    + ' is not equal to the length of lfeat_type_list, '
                    + str(len(this_lfeat_symbol)) + ' VS. '
                    + str(len(self._lfeat_type_list)))
            index = 0
            while index < len(lfeat_symbol_separate):
                lfeat_symbol_separate[index] = lfeat_symbol_separate[
                    index] + this_lfeat_symbol[index] + ' '
                index = index + 1

        index = 0
        lfeat_type = self._lfeat_type_list[index]
        sequence = self._symbols_dict.symbol_to_sequence(
            lfeat_symbol_separate[index].strip(), lfeat_type, cleaner_names)
        sequence_array = np.asarray(
            sequence[:-1],
            dtype=np.int32)  # sequence length minus 1 to ignore EOS ~
        inputs = np.eye(
            self._inputs_dim[lfeat_type], dtype=np.float32)[sequence_array]
        index = index + 1
        while index < len(self._lfeat_type_list) - 2:
            lfeat_type = self._lfeat_type_list[index]
            sequence = self._symbols_dict.symbol_to_sequence(
                lfeat_symbol_separate[index].strip(), lfeat_type,
                cleaner_names)
            sequence_array = np.asarray(
                sequence[:-1],
                dtype=np.int32)  # sequence length minus 1 to ignore EOS ~
            inputs_temp = np.eye(
                self._inputs_dim[lfeat_type], dtype=np.float32)[sequence_array]
            inputs = np.concatenate((inputs, inputs_temp), axis=1)
            index = index + 1
        seq = inputs

        lfeat_type = 'emo_category'
        inputs_emotion = multi_label_symbol_to_sequence(
            self._emo_category, lfeat_symbol_separate[index].strip())
        # inputs_emotion = inputs_emotion * 1.5
        index = index + 1

        lfeat_type = 'speaker'
        inputs_speaker = multi_label_symbol_to_sequence(
            self._speaker, lfeat_symbol_separate[index].strip())

        duration_scale = np.ones((len(seq), ), dtype=np.float32)
        start_idx = 0
        for (percent, scale) in self._duration_cfg_lst:
            duration_scale[start_idx:start_idx
                           + int(percent * len(seq))] = scale
            start_idx += int(percent * len(seq))

        pitch_contours_scale = np.ones((len(seq), ), dtype=np.float32)
        start_idx = 0
        for (percent, scale) in self._pitch_contours_cfg_lst:
            pitch_contours_scale[start_idx:start_idx
                                 + int(percent * len(seq))] = scale
            start_idx += int(percent * len(seq))

        energy_contours_scale = np.ones((len(seq), ), dtype=np.float32)
        start_idx = 0
        for (percent, scale) in self._energy_contours_cfg_lst:
            energy_contours_scale[start_idx:start_idx
                                  + int(percent * len(seq))] = scale
            start_idx += int(percent * len(seq))

        feed_dict = {
            self._model.inputs: [np.asarray(seq, dtype=np.float32)],
            self._model.inputs_emotion:
            [np.asarray(inputs_emotion, dtype=np.float32)],
            self._model.inputs_speaker:
            [np.asarray(inputs_speaker, dtype=np.float32)],
            self._model.input_lengths:
            np.asarray([len(seq)], dtype=np.int32),
            self._model.duration_scales: [duration_scale],
            self._model.pitch_scales: [pitch_contours_scale],
            self._model.energy_scales: [energy_contours_scale]
        }

        result = self._session.run([
            self._mel_spec, self._duration_outputs, self._duration_outputs_,
            self._pitch_contour_outputs, self._embedded_inputs_emotion,
            self._embedding_fsmn_outputs, self._encoder_outputs,
            self._pitch_embeddings, self._LR_outputs,
            self._postnet_fsmn_outputs, self._energy_contour_outputs,
            self._energy_embeddings, self._attention_x, self._attention_h
        ], feed_dict=feed_dict)  # yapf:disable
        return result[0]

    def vocoder_process(self, melspec):
        dim0 = list(melspec.shape)[-1]
        if dim0 != self._voc_config.num_mels:
            raise TtsVocoderMelspecShapeMismatchException(
                'input melspec mismatch require {} but {}'.format(
                    self._voc_config.num_mels, dim0))
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

    def forward(self, text):
        result = self._frontend.gen_tacotron_symbols(text)
        texts = [s for s in result.splitlines() if s != '']
        audio_total = np.empty((0), dtype='int16')
        for line in texts:
            line = line.strip().split('\t')
            audio = self.vocoder_process(
                self.am_synthesis_one_sentences(line[1]))
            audio_total = np.append(audio_total, audio, axis=0)
        return audio_total

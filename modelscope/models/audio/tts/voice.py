import os

import json
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from modelscope.utils.constant import ModelFile, Tasks
from .models import Generator, create_am_model
from .text.symbols import load_symbols
from .text.symbols_dict import SymbolsDict

import tensorflow as tf  # isort:skip

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


class Voice:

    def __init__(self, voice_name, voice_path, am_hparams, voc_config):
        self.__voice_name = voice_name
        self.__voice_path = voice_path
        self.__am_hparams = tf.contrib.training.HParams(**am_hparams)
        self.__voc_config = AttrDict(**voc_config)
        self.__model_loaded = False

    def __load_am(self):
        local_am_ckpt_path = os.path.join(self.__voice_path,
                                          ModelFile.TF_CHECKPOINT_FOLDER)
        self.__am_ckpt_path = os.path.join(local_am_ckpt_path, 'ckpt')
        self.__dict_path = os.path.join(self.__voice_path, 'dicts')
        has_mask = True
        if self.__am_hparams.get('has_mask') is not None:
            has_mask = self.__am_hparams.has_mask
        model_name = 'robutrans'
        self.__lfeat_type_list = self.__am_hparams.lfeat_type_list.strip(
        ).split(',')
        sy, tone, syllable_flag, word_segment, emo_category, speaker = load_symbols(
            self.__dict_path, has_mask)
        self.__sy = sy
        self.__tone = tone
        self.__syllable_flag = syllable_flag
        self.__word_segment = word_segment
        self.__emo_category = emo_category
        self.__speaker = speaker
        self.__inputs_dim = dict()
        for lfeat_type in self.__lfeat_type_list:
            if lfeat_type == 'sy':
                self.__inputs_dim[lfeat_type] = len(sy)
            elif lfeat_type == 'tone':
                self.__inputs_dim[lfeat_type] = len(tone)
            elif lfeat_type == 'syllable_flag':
                self.__inputs_dim[lfeat_type] = len(syllable_flag)
            elif lfeat_type == 'word_segment':
                self.__inputs_dim[lfeat_type] = len(word_segment)
            elif lfeat_type == 'emo_category':
                self.__inputs_dim[lfeat_type] = len(emo_category)
            elif lfeat_type == 'speaker':
                self.__inputs_dim[lfeat_type] = len(speaker)

        self.__symbols_dict = SymbolsDict(sy, tone, syllable_flag,
                                          word_segment, emo_category, speaker,
                                          self.__inputs_dim,
                                          self.__lfeat_type_list)
        dim_inputs = sum(self.__inputs_dim.values(
        )) - self.__inputs_dim['speaker'] - self.__inputs_dim['emo_category']
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            inputs = tf.placeholder(tf.float32, [1, None, dim_inputs],
                                    'inputs')
            inputs_emotion = tf.placeholder(
                tf.float32, [1, None, self.__inputs_dim['emo_category']],
                'inputs_emotion')
            inputs_speaker = tf.placeholder(
                tf.float32, [1, None, self.__inputs_dim['speaker']],
                'inputs_speaker')
            input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
            pitch_contours_scale = tf.placeholder(tf.float32, [1, None],
                                                  'pitch_contours_scale')
            energy_contours_scale = tf.placeholder(tf.float32, [1, None],
                                                   'energy_contours_scale')
            duration_scale = tf.placeholder(tf.float32, [1, None],
                                            'duration_scale')
            with tf.variable_scope('model') as _:
                self.__model = create_am_model(model_name, self.__am_hparams)
                self.__model.initialize(
                    inputs,
                    inputs_emotion,
                    inputs_speaker,
                    input_lengths,
                    duration_scales=duration_scale,
                    pitch_scales=pitch_contours_scale,
                    energy_scales=energy_contours_scale)
                self.__mel_spec = self.__model.mel_outputs[0]
                self.__duration_outputs = self.__model.duration_outputs[0]
                self.__duration_outputs_ = self.__model.duration_outputs_[0]
                self.__pitch_contour_outputs = self.__model.pitch_contour_outputs[
                    0]
                self.__energy_contour_outputs = self.__model.energy_contour_outputs[
                    0]
                self.__embedded_inputs_emotion = self.__model.embedded_inputs_emotion[
                    0]
                self.__embedding_fsmn_outputs = self.__model.embedding_fsmn_outputs[
                    0]
                self.__encoder_outputs = self.__model.encoder_outputs[0]
                self.__pitch_embeddings = self.__model.pitch_embeddings[0]
                self.__energy_embeddings = self.__model.energy_embeddings[0]
                self.__LR_outputs = self.__model.LR_outputs[0]
                self.__postnet_fsmn_outputs = self.__model.postnet_fsmn_outputs[
                    0]
                self.__attention_h = self.__model.attention_h
                self.__attention_x = self.__model.attention_x

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.__session = tf.Session(config=config)
                self.__session.run(tf.global_variables_initializer())

                saver = tf.train.Saver()
                saver.restore(self.__session, self.__am_ckpt_path)

    def __load_vocoder(self):
        self.__voc_ckpt_path = os.path.join(self.__voice_path,
                                            ModelFile.TORCH_MODEL_BIN_FILE)
        if torch.cuda.is_available():
            torch.manual_seed(self.__voc_config.seed)
            self.__device = torch.device('cuda')
        else:
            self.__device = torch.device('cpu')
        self.__generator = Generator(self.__voc_config).to(self.__device)
        state_dict_g = load_checkpoint(self.__voc_ckpt_path, self.__device)
        self.__generator.load_state_dict(state_dict_g['generator'])
        self.__generator.eval()
        self.__generator.remove_weight_norm()

    def __am_forward(self,
                     text,
                     pitch_control_str='',
                     duration_control_str='',
                     energy_control_str=''):
        duration_cfg_lst = []
        if len(duration_control_str) != 0:
            for item in duration_control_str.strip().split('|'):
                percent, scale = item.lstrip('(').rstrip(')').split(',')
                duration_cfg_lst.append((float(percent), float(scale)))
        pitch_contours_cfg_lst = []
        if len(pitch_control_str) != 0:
            for item in pitch_control_str.strip().split('|'):
                percent, scale = item.lstrip('(').rstrip(')').split(',')
                pitch_contours_cfg_lst.append((float(percent), float(scale)))
        energy_contours_cfg_lst = []
        if len(energy_control_str) != 0:
            for item in energy_control_str.strip().split('|'):
                percent, scale = item.lstrip('(').rstrip(')').split(',')
                energy_contours_cfg_lst.append((float(percent), float(scale)))
        cleaner_names = [
            x.strip() for x in self.__am_hparams.cleaners.split(',')
        ]

        lfeat_symbol = text.strip().split(' ')
        lfeat_symbol_separate = [''] * int(len(self.__lfeat_type_list))
        for this_lfeat_symbol in lfeat_symbol:
            this_lfeat_symbol = this_lfeat_symbol.strip('{').strip('}').split(
                '$')
            if len(this_lfeat_symbol) != len(self.__lfeat_type_list):
                raise Exception(
                    'Length of this_lfeat_symbol in training data'
                    + ' is not equal to the length of lfeat_type_list, '
                    + str(len(this_lfeat_symbol)) + ' VS. '
                    + str(len(self.__lfeat_type_list)))
            index = 0
            while index < len(lfeat_symbol_separate):
                lfeat_symbol_separate[index] = lfeat_symbol_separate[
                    index] + this_lfeat_symbol[index] + ' '
                index = index + 1

        index = 0
        lfeat_type = self.__lfeat_type_list[index]
        sequence = self.__symbols_dict.symbol_to_sequence(
            lfeat_symbol_separate[index].strip(), lfeat_type, cleaner_names)
        sequence_array = np.asarray(
            sequence[:-1],
            dtype=np.int32)  # sequence length minus 1 to ignore EOS ~
        inputs = np.eye(
            self.__inputs_dim[lfeat_type], dtype=np.float32)[sequence_array]
        index = index + 1
        while index < len(self.__lfeat_type_list) - 2:
            lfeat_type = self.__lfeat_type_list[index]
            sequence = self.__symbols_dict.symbol_to_sequence(
                lfeat_symbol_separate[index].strip(), lfeat_type,
                cleaner_names)
            sequence_array = np.asarray(
                sequence[:-1],
                dtype=np.int32)  # sequence length minus 1 to ignore EOS ~
            inputs_temp = np.eye(
                self.__inputs_dim[lfeat_type],
                dtype=np.float32)[sequence_array]
            inputs = np.concatenate((inputs, inputs_temp), axis=1)
            index = index + 1
        seq = inputs

        lfeat_type = 'emo_category'
        inputs_emotion = multi_label_symbol_to_sequence(
            self.__emo_category, lfeat_symbol_separate[index].strip())
        # inputs_emotion = inputs_emotion * 1.5
        index = index + 1

        lfeat_type = 'speaker'
        inputs_speaker = multi_label_symbol_to_sequence(
            self.__speaker, lfeat_symbol_separate[index].strip())

        duration_scale = np.ones((len(seq), ), dtype=np.float32)
        start_idx = 0
        for (percent, scale) in duration_cfg_lst:
            duration_scale[start_idx:start_idx
                           + int(percent * len(seq))] = scale
            start_idx += int(percent * len(seq))

        pitch_contours_scale = np.ones((len(seq), ), dtype=np.float32)
        start_idx = 0
        for (percent, scale) in pitch_contours_cfg_lst:
            pitch_contours_scale[start_idx:start_idx
                                 + int(percent * len(seq))] = scale
            start_idx += int(percent * len(seq))

        energy_contours_scale = np.ones((len(seq), ), dtype=np.float32)
        start_idx = 0
        for (percent, scale) in energy_contours_cfg_lst:
            energy_contours_scale[start_idx:start_idx
                                  + int(percent * len(seq))] = scale
            start_idx += int(percent * len(seq))

        feed_dict = {
            self.__model.inputs: [np.asarray(seq, dtype=np.float32)],
            self.__model.inputs_emotion:
            [np.asarray(inputs_emotion, dtype=np.float32)],
            self.__model.inputs_speaker:
            [np.asarray(inputs_speaker, dtype=np.float32)],
            self.__model.input_lengths:
            np.asarray([len(seq)], dtype=np.int32),
            self.__model.duration_scales: [duration_scale],
            self.__model.pitch_scales: [pitch_contours_scale],
            self.__model.energy_scales: [energy_contours_scale]
        }

        result = self.__session.run([
            self.__mel_spec, self.__duration_outputs, self.__duration_outputs_,
            self.__pitch_contour_outputs, self.__embedded_inputs_emotion,
            self.__embedding_fsmn_outputs, self.__encoder_outputs,
            self.__pitch_embeddings, self.__LR_outputs,
            self.__postnet_fsmn_outputs, self.__energy_contour_outputs,
            self.__energy_embeddings, self.__attention_x, self.__attention_h
        ], feed_dict=feed_dict)  # yapf:disable
        return result[0]

    def __vocoder_forward(self, melspec):
        dim0 = list(melspec.shape)[-1]
        if dim0 != self.__voc_config.num_mels:
            raise TtsVocoderMelspecShapeMismatchException(
                'input melspec mismatch require {} but {}'.format(
                    self.__voc_config.num_mels, dim0))
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

    def forward(self, text):
        if not self.__model_loaded:
            self.__load_am()
            self.__load_vocoder()
            self.__model_loaded = True
        return self.__vocoder_forward(self.__am_forward(text))

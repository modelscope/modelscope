# Copyright (c) Alibaba, Inc. and its affiliates.

import functools
import glob
import math
import os
import random
from multiprocessing import Manager

import librosa
import numpy as np
import torch
from scipy.stats import betabinom
from tqdm import tqdm

from modelscope.models.audio.tts.kantts.utils.ling_unit.ling_unit import (
    KanTtsLinguisticUnit, emotion_types)
from modelscope.utils.logger import get_logger

DATASET_RANDOM_SEED = 1234
torch.multiprocessing.set_sharing_strategy('file_system')
logging = get_logger()


@functools.lru_cache(maxsize=256)
def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


class Padder(object):

    def __init__(self):
        super(Padder, self).__init__()
        pass

    def _pad1D(self, x, length, pad):
        return np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=pad)

    def _pad2D(self, x, length, pad):
        return np.pad(
            x, [(0, length - x.shape[0]), (0, 0)],
            mode='constant',
            constant_values=pad)

    def _pad_durations(self, duration, max_in_len, max_out_len):
        framenum = np.sum(duration)
        symbolnum = duration.shape[0]
        if framenum < max_out_len:
            padframenum = max_out_len - framenum
            duration = np.insert(
                duration, symbolnum, values=padframenum, axis=0)
            duration = np.insert(
                duration,
                symbolnum + 1,
                values=[0] * (max_in_len - symbolnum - 1),
                axis=0,
            )
        else:
            if symbolnum < max_in_len:
                duration = np.insert(
                    duration,
                    symbolnum,
                    values=[0] * (max_in_len - symbolnum),
                    axis=0)
        return duration

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _prepare_scalar_inputs(self, inputs, max_len, pad):
        return torch.from_numpy(
            np.stack([self._pad1D(x, max_len, pad) for x in inputs]))

    def _prepare_targets(self, targets, max_len, pad):
        return torch.from_numpy(
            np.stack([self._pad2D(t, max_len, pad) for t in targets])).float()

    def _prepare_durations(self, durations, max_in_len, max_out_len):
        return torch.from_numpy(
            np.stack([
                self._pad_durations(t, max_in_len, max_out_len)
                for t in durations
            ])).long()


class KanttsDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        metafile,
        root_dir,
    ):
        self.meta = []
        if not isinstance(metafile, list):
            metafile = [metafile]
        if not isinstance(root_dir, list):
            root_dir = [root_dir]

        for meta_file, data_dir in zip(metafile, root_dir):
            if not os.path.exists(meta_file):
                logging.error('meta file not found: {}'.format(meta_file))
                raise ValueError(
                    '[Dataset] meta file: {} not found'.format(meta_file))
            if not os.path.exists(data_dir):
                logging.error('data directory not found: {}'.format(data_dir))
                raise ValueError(
                    '[Dataset] data dir: {} not found'.format(data_dir))
            self.meta.extend(self.load_meta(meta_file, data_dir))

    def load_meta(self, meta_file, data_dir):
        pass


class VocDataset(KanttsDataset):
    """
    provide (mel, audio) data pair
    """

    def __init__(
        self,
        metafile,
        root_dir,
        config,
    ):
        self.config = config
        self.sampling_rate = config['audio_config']['sampling_rate']
        self.n_fft = config['audio_config']['n_fft']
        self.hop_length = config['audio_config']['hop_length']
        self.batch_max_steps = config['batch_max_steps']
        self.batch_max_frames = self.batch_max_steps // self.hop_length
        self.aux_context_window = 0
        self.start_offset = self.aux_context_window
        self.end_offset = -(self.batch_max_frames + self.aux_context_window)
        self.nsf_enable = (
            config['Model']['Generator']['params'].get('nsf_params', None)
            is not None)

        super().__init__(metafile, root_dir)

        #  Load from training data directory
        if len(self.meta) == 0 and isinstance(root_dir, str):
            wav_dir = os.path.join(root_dir, 'wav')
            mel_dir = os.path.join(root_dir, 'mel')
            if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
                raise ValueError('wav or mel directory not found')
            self.meta.extend(self.load_meta_from_dir(wav_dir, mel_dir))
        elif len(self.meta) == 0 and isinstance(root_dir, list):
            for d in root_dir:
                wav_dir = os.path.join(d, 'wav')
                mel_dir = os.path.join(d, 'mel')
                if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
                    raise ValueError('wav or mel directory not found')
                self.meta.extend(self.load_meta_from_dir(wav_dir, mel_dir))

        self.allow_cache = config['allow_cache']
        if self.allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.meta))]

    @staticmethod
    def gen_metafile(wav_dir, out_dir, split_ratio=0.98):
        wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
        frame_f0_dir = os.path.join(out_dir, 'frame_f0')
        frame_uv_dir = os.path.join(out_dir, 'frame_uv')
        mel_dir = os.path.join(out_dir, 'mel')
        random.seed(DATASET_RANDOM_SEED)
        random.shuffle(wav_files)
        num_train = int(len(wav_files) * split_ratio) - 1
        with open(os.path.join(out_dir, 'train.lst'), 'w') as f:
            for wav_file in wav_files[:num_train]:
                index = os.path.splitext(os.path.basename(wav_file))[0]
                if (not os.path.exists(
                        os.path.join(frame_f0_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(frame_uv_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(mel_dir, index + '.npy'))):
                    continue
                f.write('{}\n'.format(index))

        with open(os.path.join(out_dir, 'valid.lst'), 'w') as f:
            for wav_file in wav_files[num_train:]:
                index = os.path.splitext(os.path.basename(wav_file))[0]
                if (not os.path.exists(
                        os.path.join(frame_f0_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(frame_uv_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(mel_dir, index + '.npy'))):
                    continue
                f.write('{}\n'.format(index))

    def load_meta(self, metafile, data_dir):
        with open(metafile, 'r') as f:
            lines = f.readlines()
        wav_dir = os.path.join(data_dir, 'wav')
        mel_dir = os.path.join(data_dir, 'mel')
        frame_f0_dir = os.path.join(data_dir, 'frame_f0')
        frame_uv_dir = os.path.join(data_dir, 'frame_uv')
        if not os.path.exists(wav_dir) or not os.path.exists(mel_dir):
            raise ValueError('wav or mel directory not found')
        items = []
        logging.info('Loading metafile...')
        for name in tqdm(lines):
            name = name.strip()
            mel_file = os.path.join(mel_dir, name + '.npy')
            wav_file = os.path.join(wav_dir, name + '.wav')
            frame_f0_file = os.path.join(frame_f0_dir, name + '.npy')
            frame_uv_file = os.path.join(frame_uv_dir, name + '.npy')
            items.append((wav_file, mel_file, frame_f0_file, frame_uv_file))
        return items

    def load_meta_from_dir(self, wav_dir, mel_dir):
        wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
        items = []
        for wav_file in wav_files:
            mel_file = os.path.join(mel_dir, os.path.basename(wav_file))
            if os.path.exists(mel_file):
                items.append((wav_file, mel_file))
        return items

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        wav_file, mel_file, frame_f0_file, frame_uv_file = self.meta[idx]

        wav_data = librosa.core.load(wav_file, sr=self.sampling_rate)[0]
        mel_data = np.load(mel_file)

        if self.nsf_enable:
            frame_f0_data = np.load(frame_f0_file).reshape(-1, 1)
            frame_uv_data = np.load(frame_uv_file).reshape(-1, 1)
            mel_data = np.concatenate((mel_data, frame_f0_data, frame_uv_data),
                                      axis=1)

        # make sure mel_data length greater than batch_max_frames at least 1 frame
        if mel_data.shape[0] <= self.batch_max_frames:
            mel_data = np.concatenate(
                (
                    mel_data,
                    np.zeros((
                        self.batch_max_frames - mel_data.shape[0] + 1,
                        mel_data.shape[1],
                    )),
                ),
                axis=0,
            )
            wav_cache = np.zeros(
                mel_data.shape[0] * self.hop_length, dtype=np.float32)
            wav_cache[:len(wav_data)] = wav_data
            wav_data = wav_cache
        else:
            # make sure the audio length and feature length are matched
            wav_data = np.pad(wav_data, (0, self.n_fft), mode='reflect')
            wav_data = wav_data[:len(mel_data) * self.hop_length]

        assert len(mel_data) * self.hop_length == len(wav_data)

        if self.allow_cache:
            self.caches[idx] = (wav_data, mel_data)
        return (wav_data, mel_data)

    def collate_fn(self, batch):
        wav_data, mel_data = [item[0]
                              for item in batch], [item[1] for item in batch]
        mel_lengths = [len(mel) for mel in mel_data]

        start_frames = np.array([
            np.random.randint(self.start_offset, length + self.end_offset)
            for length in mel_lengths
        ])

        wav_start = start_frames * self.hop_length
        wav_end = wav_start + self.batch_max_steps

        # aux window works as padding
        mel_start = start_frames - self.aux_context_window
        mel_end = mel_start + self.batch_max_frames + self.aux_context_window

        wav_batch = [
            x[start:end] for x, start, end in zip(wav_data, wav_start, wav_end)
        ]
        mel_batch = [
            c[start:end] for c, start, end in zip(mel_data, mel_start, mel_end)
        ]

        # (B, 1, T)
        wav_batch = torch.tensor(
            np.asarray(wav_batch), dtype=torch.float32).unsqueeze(1)
        # (B, C, T)
        mel_batch = torch.tensor(
            np.asarray(mel_batch), dtype=torch.float32).transpose(2, 1)
        return wav_batch, mel_batch


def get_voc_datasets(
    config,
    root_dir,
    split_ratio=0.98,
):
    if isinstance(root_dir, str):
        root_dir = [root_dir]
    train_meta_lst = []
    valid_meta_lst = []
    for data_dir in root_dir:
        train_meta = os.path.join(data_dir, 'train.lst')
        valid_meta = os.path.join(data_dir, 'valid.lst')
        if not os.path.exists(train_meta) or not os.path.exists(valid_meta):
            VocDataset.gen_metafile(
                os.path.join(data_dir, 'wav'), data_dir, split_ratio)
        train_meta_lst.append(train_meta)
        valid_meta_lst.append(valid_meta)
    train_dataset = VocDataset(
        train_meta_lst,
        root_dir,
        config,
    )

    valid_dataset = VocDataset(
        valid_meta_lst,
        root_dir,
        config,
    )

    return train_dataset, valid_dataset


def get_fp_label(aug_ling_txt):
    token_lst = aug_ling_txt.split(' ')
    emo_lst = [token.strip('{}').split('$')[4] for token in token_lst]
    syllable_lst = [token.strip('{}').split('$')[0] for token in token_lst]

    # EOS token append
    emo_lst.append(emotion_types[0])
    syllable_lst.append('EOS')

    # According to the original emotion tag, set each token's fp label.
    if emo_lst[0] != emotion_types[3]:
        emo_lst[0] = emotion_types[0]
        emo_lst[1] = emotion_types[0]
    for i in range(len(emo_lst) - 2, 1, -1):
        if emo_lst[i] != emotion_types[3] and emo_lst[i
                                                      - 1] != emotion_types[3]:
            emo_lst[i] = emotion_types[0]
        elif emo_lst[i] != emotion_types[3] and emo_lst[
                i - 1] == emotion_types[3]:
            emo_lst[i] = emotion_types[3]
            if syllable_lst[i - 2] == 'ga':
                emo_lst[i + 1] = emotion_types[1]
            elif syllable_lst[i - 2] == 'ge' and syllable_lst[i - 1] == 'en_c':
                emo_lst[i + 1] = emotion_types[2]
            else:
                emo_lst[i + 1] = emotion_types[4]

    fp_label = []
    for i in range(len(emo_lst)):
        if emo_lst[i] == emotion_types[0]:
            fp_label.append(0)
        elif emo_lst[i] == emotion_types[1]:
            fp_label.append(1)
        elif emo_lst[i] == emotion_types[2]:
            fp_label.append(2)
        elif emo_lst[i] == emotion_types[3]:
            continue
        elif emo_lst[i] == emotion_types[4]:
            fp_label.append(3)
        else:
            pass

    return np.array(fp_label)


class AmDataset(KanttsDataset):
    """
    provide (ling, emo, speaker, mel) pair
    """

    def __init__(
        self,
        metafile,
        root_dir,
        config,
        lang_dir=None,
        allow_cache=False,
    ):
        self.config = config
        self.with_duration = True
        self.nsf_enable = self.config['Model']['KanTtsSAMBERT']['params'].get(
            'NSF', False)
        self.fp_enable = self.config['Model']['KanTtsSAMBERT']['params'].get(
            'FP', False)

        super().__init__(metafile, root_dir)
        self.allow_cache = allow_cache

        self.ling_unit = KanTtsLinguisticUnit(config, lang_dir)
        self.padder = Padder()

        self.r = self.config['Model']['KanTtsSAMBERT']['params'][
            'outputs_per_step']

        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.meta))]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        (
            ling_txt,
            mel_file,
            dur_file,
            f0_file,
            energy_file,
            frame_f0_file,
            frame_uv_file,
            aug_ling_txt,
        ) = self.meta[idx]

        ling_data = self.ling_unit.encode_symbol_sequence(ling_txt)
        mel_data = np.load(mel_file)
        dur_data = np.load(dur_file) if dur_file is not None else None
        f0_data = np.load(f0_file)
        energy_data = np.load(energy_file)

        # generate fp position label according to fpadd_meta
        if self.fp_enable and aug_ling_txt is not None:
            fp_label = get_fp_label(aug_ling_txt)
        else:
            fp_label = None

        if self.with_duration:
            attn_prior = None
        else:
            attn_prior = beta_binomial_prior_distribution(
                len(ling_data[0]), mel_data.shape[0])

        # Concat frame-level f0 and uv to mel_data
        if self.nsf_enable:
            frame_f0_data = np.load(frame_f0_file).reshape(-1, 1)
            frame_uv_data = np.load(frame_uv_file).reshape(-1, 1)
            mel_data = np.concatenate([mel_data, frame_f0_data, frame_uv_data],
                                      axis=1)

        if self.allow_cache:
            self.caches[idx] = (
                ling_data,
                mel_data,
                dur_data,
                f0_data,
                energy_data,
                attn_prior,
                fp_label,
            )

        return (
            ling_data,
            mel_data,
            dur_data,
            f0_data,
            energy_data,
            attn_prior,
            fp_label,
        )

    def load_meta(self, metafile, data_dir):
        with open(metafile, 'r') as f:
            lines = f.readlines()

        aug_ling_dict = {}
        if self.fp_enable:
            add_fp_metafile = metafile.replace('fprm', 'fpadd')
            with open(add_fp_metafile, 'r') as f:
                fpadd_lines = f.readlines()
            for line in fpadd_lines:
                index, aug_ling_txt = line.split('\t')
                aug_ling_dict[index] = aug_ling_txt

        mel_dir = os.path.join(data_dir, 'mel')
        dur_dir = os.path.join(data_dir, 'duration')
        f0_dir = os.path.join(data_dir, 'f0')
        energy_dir = os.path.join(data_dir, 'energy')
        frame_f0_dir = os.path.join(data_dir, 'frame_f0')
        frame_uv_dir = os.path.join(data_dir, 'frame_uv')

        self.with_duration = os.path.exists(dur_dir)

        items = []
        logging.info('Loading metafile...')
        for line in tqdm(lines):
            line = line.strip()
            index, ling_txt = line.split('\t')
            mel_file = os.path.join(mel_dir, index + '.npy')
            if self.with_duration:
                dur_file = os.path.join(dur_dir, index + '.npy')
            else:
                dur_file = None
            f0_file = os.path.join(f0_dir, index + '.npy')
            energy_file = os.path.join(energy_dir, index + '.npy')
            frame_f0_file = os.path.join(frame_f0_dir, index + '.npy')
            frame_uv_file = os.path.join(frame_uv_dir, index + '.npy')
            aug_ling_txt = aug_ling_dict.get(index, None)
            if self.fp_enable and aug_ling_txt is None:
                logging.warning(f'Missing fpadd meta for {index}')
                continue

            items.append((
                ling_txt,
                mel_file,
                dur_file,
                f0_file,
                energy_file,
                frame_f0_file,
                frame_uv_file,
                aug_ling_txt,
            ))

        return items

    def load_fpadd_meta(self, metafile):
        with open(metafile, 'r') as f:
            lines = f.readlines()

        items = []
        logging.info('Loading fpadd metafile...')
        for line in tqdm(lines):
            line = line.strip()
            index, ling_txt = line.split('\t')

            items.append((ling_txt, ))

        return items

    @staticmethod
    def gen_metafile(
        raw_meta_file,
        out_dir,
        train_meta_file,
        valid_meta_file,
        badlist=None,
        split_ratio=0.98,
    ):
        with open(raw_meta_file, 'r') as f:
            lines = f.readlines()
        frame_f0_dir = os.path.join(out_dir, 'frame_f0')
        frame_uv_dir = os.path.join(out_dir, 'frame_uv')
        mel_dir = os.path.join(out_dir, 'mel')
        duration_dir = os.path.join(out_dir, 'duration')
        random.seed(DATASET_RANDOM_SEED)
        random.shuffle(lines)
        num_train = int(len(lines) * split_ratio) - 1
        with open(train_meta_file, 'w') as f:
            for line in lines[:num_train]:
                index = line.split('\t')[0]
                if badlist is not None and index in badlist:
                    continue
                if (not os.path.exists(
                        os.path.join(frame_f0_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(frame_uv_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(mel_dir, index + '.npy'))):
                    continue
                if os.path.exists(duration_dir) and not os.path.exists(
                        os.path.join(duration_dir, index + '.npy')):
                    continue
                f.write(line)

        with open(valid_meta_file, 'w') as f:
            for line in lines[num_train:]:
                index = line.split('\t')[0]
                if badlist is not None and index in badlist:
                    continue
                if (not os.path.exists(
                        os.path.join(frame_f0_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(frame_uv_dir, index + '.npy'))
                        or not os.path.exists(
                            os.path.join(mel_dir, index + '.npy'))):
                    continue
                if os.path.exists(duration_dir) and not os.path.exists(
                        os.path.join(duration_dir, index + '.npy')):
                    continue
                f.write(line)

    def collate_fn(self, batch):
        data_dict = {}

        max_input_length = max((len(x[0][0]) for x in batch))
        if self.with_duration:
            max_dur_length = max((x[2].shape[0] for x in batch)) + 1

        lfeat_type_index = 0
        lfeat_type = self.ling_unit._lfeat_type_list[lfeat_type_index]
        if self.ling_unit.using_byte():
            # for byte-based model only
            inputs_byte_index = self.padder._prepare_scalar_inputs(
                [x[0][lfeat_type_index] for x in batch],
                max_input_length,
                self.ling_unit._sub_unit_pad[lfeat_type],
            ).long()

            data_dict['input_lings'] = torch.stack([inputs_byte_index], dim=2)
        else:
            # pure linguistic info: sy|tone|syllable_flag|word_segment
            # sy
            inputs_sy = self.padder._prepare_scalar_inputs(
                [x[0][lfeat_type_index] for x in batch],
                max_input_length,
                self.ling_unit._sub_unit_pad[lfeat_type],
            ).long()

            # tone
            lfeat_type_index = lfeat_type_index + 1
            lfeat_type = self.ling_unit._lfeat_type_list[lfeat_type_index]
            inputs_tone = self.padder._prepare_scalar_inputs(
                [x[0][lfeat_type_index] for x in batch],
                max_input_length,
                self.ling_unit._sub_unit_pad[lfeat_type],
            ).long()

            # syllable_flag
            lfeat_type_index = lfeat_type_index + 1
            lfeat_type = self.ling_unit._lfeat_type_list[lfeat_type_index]
            inputs_syllable_flag = self.padder._prepare_scalar_inputs(
                [x[0][lfeat_type_index] for x in batch],
                max_input_length,
                self.ling_unit._sub_unit_pad[lfeat_type],
            ).long()

            # word_segment
            lfeat_type_index = lfeat_type_index + 1
            lfeat_type = self.ling_unit._lfeat_type_list[lfeat_type_index]
            inputs_ws = self.padder._prepare_scalar_inputs(
                [x[0][lfeat_type_index] for x in batch],
                max_input_length,
                self.ling_unit._sub_unit_pad[lfeat_type],
            ).long()

            data_dict['input_lings'] = torch.stack(
                [inputs_sy, inputs_tone, inputs_syllable_flag, inputs_ws],
                dim=2)

        # emotion category
        lfeat_type_index = lfeat_type_index + 1
        lfeat_type = self.ling_unit._lfeat_type_list[lfeat_type_index]
        data_dict['input_emotions'] = self.padder._prepare_scalar_inputs(
            [x[0][lfeat_type_index] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # speaker category
        lfeat_type_index = lfeat_type_index + 1
        lfeat_type = self.ling_unit._lfeat_type_list[lfeat_type_index]
        data_dict['input_speakers'] = self.padder._prepare_scalar_inputs(
            [x[0][lfeat_type_index] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # fp label category
        if self.fp_enable:
            data_dict['fp_label'] = self.padder._prepare_scalar_inputs(
                [x[6] for x in batch],
                max_input_length,
                0,
            ).long()

        data_dict['valid_input_lengths'] = torch.as_tensor(
            [len(x[0][0]) - 1 for x in batch], dtype=torch.long
        )  # 输入的symbol sequence会在后面拼一个“~”，影响duration计算，所以把length-1
        data_dict['valid_output_lengths'] = torch.as_tensor(
            [len(x[1]) for x in batch], dtype=torch.long)

        max_output_length = torch.max(data_dict['valid_output_lengths']).item()
        max_output_round_length = self.padder._round_up(
            max_output_length, self.r)

        data_dict['mel_targets'] = self.padder._prepare_targets(
            [x[1] for x in batch], max_output_round_length, 0.0)
        if self.with_duration:
            data_dict['durations'] = self.padder._prepare_durations(
                [x[2] for x in batch], max_dur_length, max_output_round_length)
        else:
            data_dict['durations'] = None

        if self.with_duration:
            if self.fp_enable:
                feats_padding_length = max_dur_length
            else:
                feats_padding_length = max_input_length
        else:
            feats_padding_length = max_output_round_length

        data_dict['pitch_contours'] = self.padder._prepare_scalar_inputs(
            [x[3] for x in batch], feats_padding_length, 0.0).float()
        data_dict['energy_contours'] = self.padder._prepare_scalar_inputs(
            [x[4] for x in batch], feats_padding_length, 0.0).float()

        if self.with_duration:
            data_dict['attn_priors'] = None
        else:
            data_dict['attn_priors'] = torch.zeros(
                len(batch), max_output_round_length, max_input_length)
            for i in range(len(batch)):
                attn_prior = batch[i][5]
                data_dict['attn_priors'][
                    i, :attn_prior.shape[0], :attn_prior.shape[1]] = attn_prior
        return data_dict


def get_am_datasets(
    metafile,
    root_dir,
    lang_dir,
    config,
    allow_cache,
    split_ratio=0.98,
):
    if not isinstance(root_dir, list):
        root_dir = [root_dir]
    if not isinstance(metafile, list):
        metafile = [metafile]

    train_meta_lst = []
    valid_meta_lst = []

    fp_enable = config['Model']['KanTtsSAMBERT']['params'].get('FP', False)

    if fp_enable:
        am_train_fn = 'am_fprm_train.lst'
        am_valid_fn = 'am_fprm_valid.lst'
    else:
        am_train_fn = 'am_train.lst'
        am_valid_fn = 'am_valid.lst'

    for raw_metafile, data_dir in zip(metafile, root_dir):
        train_meta = os.path.join(data_dir, am_train_fn)
        valid_meta = os.path.join(data_dir, am_valid_fn)
        if not os.path.exists(train_meta) or not os.path.exists(valid_meta):
            AmDataset.gen_metafile(raw_metafile, data_dir, train_meta,
                                   valid_meta, split_ratio)
        train_meta_lst.append(train_meta)
        valid_meta_lst.append(valid_meta)

    train_dataset = AmDataset(train_meta_lst, root_dir, config, lang_dir,
                              allow_cache)

    valid_dataset = AmDataset(valid_meta_lst, root_dir, config, lang_dir,
                              allow_cache)

    return train_dataset, valid_dataset


class MaskingActor(object):

    def __init__(self, mask_ratio=0.15):
        super(MaskingActor, self).__init__()
        self.mask_ratio = mask_ratio
        pass

    def _get_random_mask(self, length, p1=0.15):
        mask = np.random.uniform(0, 1, length)
        index = 0
        while index < len(mask):
            if mask[index] < p1:
                mask[index] = 1
            else:
                mask[index] = 0
            index += 1

        return mask

    def _input_bert_masking(
        self,
        sequence_array,
        nb_symbol_category,
        mask_symbol_id,
        mask,
        p2=0.8,
        p3=0.1,
        p4=0.1,
    ):
        sequence_array_mask = sequence_array.copy()
        mask_id = np.where(mask == 1)[0]
        mask_len = len(mask_id)
        rand = np.arange(mask_len)
        np.random.shuffle(rand)

        # [MASK]
        mask_id_p2 = mask_id[rand[0:int(math.floor(mask_len * p2))]]
        if len(mask_id_p2) > 0:
            sequence_array_mask[mask_id_p2] = mask_symbol_id

        # rand
        mask_id_p3 = mask_id[
            rand[int(math.floor(mask_len * p2)):int(math.floor(mask_len * p2))
                 + int(math.floor(mask_len * p3))]]
        if len(mask_id_p3) > 0:
            sequence_array_mask[mask_id_p3] = random.randint(
                0, nb_symbol_category - 1)

        # ori
        # do nothing

        return sequence_array_mask


class BERTTextDataset(torch.utils.data.Dataset):
    """
    provide (ling, ling_sy_masked, bert_mask) pair
    """

    def __init__(
        self,
        config,
        metafile,
        root_dir,
        lang_dir=None,
        allow_cache=False,
    ):
        self.meta = []
        self.config = config

        if not isinstance(metafile, list):
            metafile = [metafile]
        if not isinstance(root_dir, list):
            root_dir = [root_dir]

        for meta_file, data_dir in zip(metafile, root_dir):
            if not os.path.exists(meta_file):
                logging.error('meta file not found: {}'.format(meta_file))
                raise ValueError(
                    '[BERT_Text_Dataset] meta file: {} not found'.format(
                        meta_file))
            if not os.path.exists(data_dir):
                logging.error('data dir not found: {}'.format(data_dir))
                raise ValueError(
                    '[BERT_Text_Dataset] data dir: {} not found'.format(
                        data_dir))
            self.meta.extend(self.load_meta(meta_file, data_dir))

        self.allow_cache = allow_cache

        self.ling_unit = KanTtsLinguisticUnit(config, lang_dir)
        self.padder = Padder()
        self.masking_actor = MaskingActor(
            self.config['Model']['KanTtsTextsyBERT']['params']['mask_ratio'])

        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.meta))]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            ling_data = self.caches[idx][0]
            bert_mask, ling_sy_masked_data = self.bert_masking(ling_data)
            return (ling_data, ling_sy_masked_data, bert_mask)

        ling_txt = self.meta[idx]

        ling_data = self.ling_unit.encode_symbol_sequence(ling_txt)
        bert_mask, ling_sy_masked_data = self.bert_masking(ling_data)

        if self.allow_cache:
            self.caches[idx] = (ling_data, )

        return (ling_data, ling_sy_masked_data, bert_mask)

    def load_meta(self, metafile, data_dir):
        with open(metafile, 'r') as f:
            lines = f.readlines()

        items = []
        logging.info('Loading metafile...')
        for line in tqdm(lines):
            line = line.strip()
            index, ling_txt = line.split('\t')

            items.append((ling_txt))

        return items

    @staticmethod
    def gen_metafile(raw_meta_file, out_dir, split_ratio=0.98):
        with open(raw_meta_file, 'r') as f:
            lines = f.readlines()
        random.seed(DATASET_RANDOM_SEED)
        random.shuffle(lines)
        num_train = int(len(lines) * split_ratio) - 1
        with open(os.path.join(out_dir, 'bert_train.lst'), 'w') as f:
            for line in lines[:num_train]:
                f.write(line)

        with open(os.path.join(out_dir, 'bert_valid.lst'), 'w') as f:
            for line in lines[num_train:]:
                f.write(line)

    def bert_masking(self, ling_data):
        length = len(ling_data[0])
        mask = self.masking_actor._get_random_mask(
            length, p1=self.masking_actor.mask_ratio)
        mask[-1] = 0

        # sy_masked
        sy_mask_symbol_id = self.ling_unit.encode_sy([self.ling_unit._mask])[0]
        ling_sy_masked_data = self.masking_actor._input_bert_masking(
            ling_data[0],
            self.ling_unit.get_unit_size()['sy'],
            sy_mask_symbol_id,
            mask,
            p2=0.8,
            p3=0.1,
            p4=0.1,
        )

        return (mask, ling_sy_masked_data)

    def collate_fn(self, batch):
        data_dict = {}

        max_input_length = max((len(x[0][0]) for x in batch))

        # pure linguistic info: sy|tone|syllable_flag|word_segment
        # sy
        lfeat_type = self.ling_unit._lfeat_type_list[0]
        targets_sy = self.padder._prepare_scalar_inputs(
            [x[0][0] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()
        # sy masked
        inputs_sy = self.padder._prepare_scalar_inputs(
            [x[1] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()
        # tone
        lfeat_type = self.ling_unit._lfeat_type_list[1]
        inputs_tone = self.padder._prepare_scalar_inputs(
            [x[0][1] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # syllable_flag
        lfeat_type = self.ling_unit._lfeat_type_list[2]
        inputs_syllable_flag = self.padder._prepare_scalar_inputs(
            [x[0][2] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        # word_segment
        lfeat_type = self.ling_unit._lfeat_type_list[3]
        inputs_ws = self.padder._prepare_scalar_inputs(
            [x[0][3] for x in batch],
            max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type],
        ).long()

        data_dict['input_lings'] = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable_flag, inputs_ws], dim=2)
        data_dict['valid_input_lengths'] = torch.as_tensor(
            [len(x[0][0]) - 1 for x in batch], dtype=torch.long
        )  # 输入的symbol sequence会在后面拼一个“~”，影响duration计算，所以把length-1

        data_dict['targets'] = targets_sy
        data_dict['bert_masks'] = self.padder._prepare_scalar_inputs(
            [x[2] for x in batch], max_input_length, 0.0)

        return data_dict


def get_bert_text_datasets(
    metafile,
    root_dir,
    config,
    allow_cache,
    split_ratio=0.98,
):
    if not isinstance(root_dir, list):
        root_dir = [root_dir]
    if not isinstance(metafile, list):
        metafile = [metafile]

    train_meta_lst = []
    valid_meta_lst = []

    for raw_metafile, data_dir in zip(metafile, root_dir):
        train_meta = os.path.join(data_dir, 'bert_train.lst')
        valid_meta = os.path.join(data_dir, 'bert_valid.lst')
        if not os.path.exists(train_meta) or not os.path.exists(valid_meta):
            BERTTextDataset.gen_metafile(raw_metafile, data_dir, split_ratio)
        train_meta_lst.append(train_meta)
        valid_meta_lst.append(valid_meta)

    train_dataset = BERTTextDataset(config, train_meta_lst, root_dir,
                                    allow_cache)

    valid_dataset = BERTTextDataset(config, valid_meta_lst, root_dir,
                                    allow_cache)

    return train_dataset, valid_dataset

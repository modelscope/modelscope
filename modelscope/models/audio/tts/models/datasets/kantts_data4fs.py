# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from modelscope.utils.logger import get_logger
from .units import KanTtsLinguisticUnit

logger = get_logger()


class KanTtsText2MelDataset(Dataset):

    def __init__(self, metadata_filename, config_filename, cache=False):
        super(KanTtsText2MelDataset, self).__init__()

        self.cache = cache

        with open(config_filename) as f:
            self._config = json.loads(f.read())

        # Load metadata:
        self._datadir = os.path.dirname(metadata_filename)
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            self._length_lst = [int(x[2]) for x in self._metadata]
            hours = sum(
                self._length_lst) * self._config['audio']['frame_shift_ms'] / (
                    3600 * 1000)

            logger.info('Loaded metadata for %d examples (%.2f hours)' %
                        (len(self._metadata), hours))
            logger.info('Minimum length: %d, Maximum length: %d' %
                        (min(self._length_lst), max(self._length_lst)))

        self.ling_unit = KanTtsLinguisticUnit(config_filename)
        self.pad_executor = KanTtsText2MelPad()

        self.r = self._config['am']['outputs_per_step']
        self.num_mels = self._config['am']['num_mels']

        if 'adv' in self._config:
            self.feat_window = self._config['adv']['random_window']
        else:
            self.feat_window = None
        logger.info(self.feat_window)

        self.data_cache = [
            self.cache_load(i) for i in tqdm(range(self.__len__()))
        ] if self.cache else []

    def get_frames_lst(self):
        return self._length_lst

    def __getitem__(self, index):
        if self.cache:
            sample = self.data_cache[index]
            return sample

        return self.cache_load(index)

    def cache_load(self, index):
        sample = {}

        meta = self._metadata[index]

        sample['utt_id'] = meta[0]

        sample['mel_target'] = np.load(os.path.join(
            self._datadir, meta[1]))[:, :self.num_mels]
        sample['output_length'] = len(sample['mel_target'])

        lfeat_symbol = meta[3]
        sample['ling'] = self.ling_unit.encode_symbol_sequence(lfeat_symbol)

        sample['duration'] = np.load(os.path.join(self._datadir, meta[4]))

        sample['pitch_contour'] = np.load(os.path.join(self._datadir, meta[5]))

        sample['energy_contour'] = np.load(
            os.path.join(self._datadir, meta[6]))

        return sample

    def __len__(self):
        return len(self._metadata)

    def collate_fn(self, batch):
        data_dict = {}

        max_input_length = max((len(x['ling'][0]) for x in batch))

        # pure linguistic info: sy|tone|syllable_flag|word_segment

        # sy
        lfeat_type = self.ling_unit._lfeat_type_list[0]
        inputs_sy = self.pad_executor._prepare_scalar_inputs(
            [x['ling'][0] for x in batch], max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type]).long()
        # tone
        lfeat_type = self.ling_unit._lfeat_type_list[1]
        inputs_tone = self.pad_executor._prepare_scalar_inputs(
            [x['ling'][1] for x in batch], max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type]).long()

        # syllable_flag
        lfeat_type = self.ling_unit._lfeat_type_list[2]
        inputs_syllable_flag = self.pad_executor._prepare_scalar_inputs(
            [x['ling'][2] for x in batch], max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type]).long()

        # word_segment
        lfeat_type = self.ling_unit._lfeat_type_list[3]
        inputs_ws = self.pad_executor._prepare_scalar_inputs(
            [x['ling'][3] for x in batch], max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type]).long()

        # emotion category
        lfeat_type = self.ling_unit._lfeat_type_list[4]
        data_dict['input_emotions'] = self.pad_executor._prepare_scalar_inputs(
            [x['ling'][4] for x in batch], max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type]).long()

        # speaker category
        lfeat_type = self.ling_unit._lfeat_type_list[5]
        data_dict['input_speakers'] = self.pad_executor._prepare_scalar_inputs(
            [x['ling'][5] for x in batch], max_input_length,
            self.ling_unit._sub_unit_pad[lfeat_type]).long()

        data_dict['input_lings'] = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable_flag, inputs_ws], dim=2)

        data_dict['valid_input_lengths'] = torch.as_tensor(
            [len(x['ling'][0]) - 1 for x in batch], dtype=torch.long
        )  # There is one '~' in the last of symbol sequence. We put length-1 for calculation.

        data_dict['valid_output_lengths'] = torch.as_tensor(
            [x['output_length'] for x in batch], dtype=torch.long)
        max_output_length = torch.max(data_dict['valid_output_lengths']).item()
        max_output_round_length = self.pad_executor._round_up(
            max_output_length, self.r)

        if self.feat_window is not None:
            active_feat_len = np.minimum(max_output_round_length,
                                         self.feat_window)
            if active_feat_len < self.feat_window:
                max_output_round_length = self.pad_executor._round_up(
                    self.feat_window, self.r)
                active_feat_len = self.feat_window

            max_offsets = [x['output_length'] - active_feat_len for x in batch]
            feat_offsets = [
                np.random.randint(0, np.maximum(1, offset))
                for offset in max_offsets
            ]
            feat_offsets = torch.from_numpy(
                np.asarray(feat_offsets, dtype=np.int32)).long()
            data_dict['feat_offsets'] = feat_offsets

        data_dict['mel_targets'] = self.pad_executor._prepare_targets(
            [x['mel_target'] for x in batch], max_output_round_length, 0.0)
        data_dict['durations'] = self.pad_executor._prepare_durations(
            [x['duration'] for x in batch], max_input_length,
            max_output_round_length)

        data_dict['pitch_contours'] = self.pad_executor._prepare_scalar_inputs(
            [x['pitch_contour'] for x in batch], max_input_length,
            0.0).float()
        data_dict[
            'energy_contours'] = self.pad_executor._prepare_scalar_inputs(
                [x['energy_contour'] for x in batch], max_input_length,
                0.0).float()

        data_dict['utt_ids'] = [x['utt_id'] for x in batch]

        return data_dict


class KanTtsText2MelPad(object):

    def __init__(self):
        super(KanTtsText2MelPad, self).__init__()
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
                axis=0)
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

# Copyright (c) 2021 Binbin Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import json
import kaldiio
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

# torch.set_printoptions(profile="full")
from modelscope.utils.logger import get_logger

logger = get_logger()


def parse_wav(data):
    """ Parse key/wav/txt from dict line

        Args:
            data: Iterable[dict()], dict has key/wav/txt/sample_rate keys

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        obj = sample['src']
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']

        try:
            sample_rate, kaldi_waveform = kaldiio.load_mat(wav_file)
            waveform = torch.tensor(kaldi_waveform, dtype=torch.float32)
            waveform = waveform.unsqueeze(0)
            example = dict(
                key=key, label=txt, wav=waveform, sample_rate=sample_rate)
            yield example
        except Exception:
            logger.warning('Failed to read {}'.format(wav_file))


def filter(data, max_length=10240, min_length=10):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample or 'feat' in sample
        num_frames = -1
        if 'wav' in sample:
            # sample['wav'] is torch.Tensor, we have 100 frames every second
            num_frames = int(sample['wav'].size(1) / sample['sample_rate']
                             * 100)
        elif 'feat' in sample:
            num_frames = sample['feat'].size(0)

        # print("{} num frames is {}".format(sample['key'], num_frames))
        if num_frames < min_length:
            logger.warning('{} is discard for too short: {} frames'.format(
                sample['key'], num_frames))
            continue
        if num_frames > max_length:
            logger.warning('{} is discard for too long: {} frames'.format(
                sample['key'], num_frames))
            continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        if 'wav' in sample:
            sample_rate = sample['sample_rate']
            waveform = sample['wav']
            if sample_rate != resample_rate:
                sample['sample_rate'] = resample_rate
                sample['wav'] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(
                        waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample


def compute_mfcc(
    data,
    feature_type='mfcc',
    num_ceps=80,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
):
    """Extract mfcc

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        # waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(
            waveform,
            num_ceps=num_ceps,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            energy_floor=0.0,
            sample_frequency=sample_rate,
        )
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def compute_fbank(data,
                  feature_type='fbank',
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        # waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            energy_floor=0.0,
            window_type='hamming',
            sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=1000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def context_expansion(data, left=1, right=1):
    """ expand left and right frames
        Args:
            data: Iterable[{key, feat, label}]
            left (int): feature left context frames
            right (int): feature right context frames

        Returns:
            data: Iterable[{key, feat, label}]
    """
    for sample in data:
        index = 0
        feats = sample['feat']
        ctx_dim = feats.shape[0]
        ctx_frm = feats.shape[1] * (left + right + 1)
        feats_ctx = torch.zeros(ctx_dim, ctx_frm, dtype=torch.float32)
        for lag in range(-left, right + 1):
            feats_ctx[:, index:index + feats.shape[1]] = torch.roll(
                feats, -lag, 0)
            index = index + feats.shape[1]

        # replication pad left margin
        for idx in range(left):
            for cpx in range(left - idx):
                feats_ctx[idx, cpx * feats.shape[1]:(cpx + 1)
                          * feats.shape[1]] = feats_ctx[left, :feats.shape[1]]

        feats_ctx = feats_ctx[:feats_ctx.shape[0] - right]
        sample['feat'] = feats_ctx
        yield sample


def frame_skip(data, skip_rate=1):
    """ skip frame
        Args:
            data: Iterable[{key, feat, label}]
            skip_rate (int): take every N-frames for model input

        Returns:
            data: Iterable[{key, feat, label}]
    """
    for sample in data:
        feats_skip = sample['feat'][::skip_rate, :]
        sample['feat'] = feats_skip
        yield sample


def batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]

        assert type(sample[0]['label']) is list

        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int32) for i in order
        ]
        label_lengths = torch.tensor([len(sample[i]['label']) for i in order],
                                     dtype=torch.int32)

        padded_feats = pad_sequence(
            sorted_feats, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(
            sorted_labels, batch_first=True, padding_value=-1)
        yield (sorted_keys, padded_feats, padded_labels, feats_lengths,
               label_lengths)

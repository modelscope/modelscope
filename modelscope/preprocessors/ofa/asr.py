# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
from pathlib import Path
from typing import Any, Dict

import librosa
import soundfile as sf
import torch
from fairseq.data.audio.feature_transforms import \
    CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig

from modelscope.utils.chinese_utils import pre_chinese
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor
from .utils.text2phone import Text2Phone


class OfaASRPreprocessor(OfaBasePreprocessor):

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaASRPreprocessor, self).__init__(cfg, model_dir, mode, *args,
                                                 **kwargs)
        # Initialize transform
        self.data_cfg = S2TDataConfig(
            Path(os.path.join(model_dir, 'fbank_config.yaml')))
        self.train_audio_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms('train', True))
        self.test_audio_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms('test', False))
        self.text2phone_tokenizer = Text2Phone(
            os.path.join(model_dir, 'text2phone_dict.txt'))
        self.phone_to_id, self.id_to_phone = self.build_phone_dict(
            os.path.join(model_dir, 'phone_dict.txt'))

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        speed = random.choice([0.9, 1.0, 1.1])
        audio_bytes = self.get_audio_bytes(data[self.column_map['wav']])
        wav, sr = librosa.load(audio_bytes, 16000, mono=True)
        fbank = self.prepare_fbank(
            torch.tensor([wav], dtype=torch.float32),
            sr,
            speed,
            target_sample_rate=16000,
            is_train=True)
        fbank_mask = torch.tensor([True])
        sample = {
            'fbank': fbank,
            'fbank_mask': fbank_mask,
            'label': data[self.column_map['text']]
        }

        target = sample['label']
        if self.language == 'zh':
            target = pre_chinese(target, self.max_tgt_length)
            sample['target'] = self.tokenize_text(target, add_bos=False)
        else:
            target = target.translate(self.transtab).strip()
            target_token_list = target.strip().split()
            target = ' '.join(target_token_list[:self.max_tgt_length])
            sample['target'] = self.tokenize_text(target, add_bos=False)

        phone_item = self.to_phone(target) + 1
        phone_mask = torch.tensor([False])

        sample['phone_item'] = phone_item + 3
        sample['phone_target'] = phone_item
        sample['phone_mask'] = phone_mask

        sample['prev_output_tokens'] = torch.cat(
            [self.bos_item, sample['target'][:-1]])
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        speed = 1.0
        audio_bytes = self.get_audio_bytes(data[self.column_map['wav']])
        wav, sr = librosa.load(audio_bytes, 16000, mono=True)
        fbank = self.prepare_fbank(
            torch.tensor([wav], dtype=torch.float32),
            sr,
            speed,
            target_sample_rate=16000,
            is_train=False)
        fbank_mask = torch.tensor([True])

        sample = {'fbank': fbank, 'fbank_mask': fbank_mask}

        if 'text' in self.column_map and self.column_map['text'] in data:
            sample['label'] = data[self.column_map['text']]

        # mock
        sample['phone_item'] = torch.tensor([6, 6, 6])
        sample['phone_mask'] = torch.tensor([False])

        return sample

    def to_phone(self, text):
        phones = self.text2phone_tokenizer.trans(text)
        ids = torch.tensor([self.phone_to_id[x] for x in phones.split(' ')])
        return ids

    def build_phone_dict(self, phone_dict_path):
        phone_to_id = dict()
        id_to_phone = dict()
        with open(phone_dict_path, 'r') as phone_dict_file:
            for i, line in enumerate(phone_dict_file):
                phone = line.strip().split(' ')[0]
                phone_to_id[phone] = i
                id_to_phone[i] = phone_to_id
        return phone_to_id, id_to_phone

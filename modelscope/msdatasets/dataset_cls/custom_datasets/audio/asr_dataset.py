# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.utils.constant import DownloadMode


class ASRDataset(MsDataset):
    """ASR dataset for speech recognition.
    support load dataset from msdataset hub or local data_dir (including wav.scp and text)
    For more details, please refer to
        https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/datasets/ms_dataset.py.
    """

    @classmethod
    def load_core(cls, data_dir, data_set):
        wav_file = os.path.join(data_dir, data_set, 'wav.scp')
        text_file = os.path.join(data_dir, data_set, 'text')
        with open(wav_file) as f:
            wav_lines = f.readlines()
        with open(text_file) as f:
            text_lines = f.readlines()
        data_list = []
        for wav_line, text_line in zip(wav_lines, text_lines):
            item = {}
            item['Audio:FILE'] = wav_line.strip().split()[-1]
            item['Text:LABEL'] = ' '.join(text_line.strip().split()[1:])
            data_list.append(item)
        return data_list

    @classmethod
    def load(
        cls,
        dataset_name,
        namespace='speech_asr',
        train_set='train',
        dev_set='validation',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    ):
        if os.path.exists(dataset_name):
            data_dir = dataset_name
            ds_dict = {}
            ds_dict['train'] = cls.load_core(data_dir, train_set)
            ds_dict['validation'] = cls.load_core(data_dir, dev_set)
            ds_dict['raw_data_dir'] = data_dir
            return ds_dict
        else:
            from modelscope.msdatasets import MsDataset

            ds_dict = MsDataset.load(
                dataset_name=dataset_name,
                namespace=namespace,
                download_mode=download_mode,
            )
            return ds_dict

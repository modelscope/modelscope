# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
from os import path as osp

from torch.utils.data import Dataset

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile, Tasks
from .ofa_file_dataset import OFAFileDataset


class OFADataset(Dataset):

    def __init__(self,
                 model_dir,
                 file_path,
                 dtypes=None,
                 separator='\t',
                 cached_index=False,
                 split=ModeKeys.TRAIN,
                 **kwargs):
        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        selected_col_ids = self.cfg.dataset.selected_col_ids
        selected_col_keys = self.cfg.dataset.selected_col_keys

        assert selected_col_ids is not None
        assert selected_col_keys is not None
        self.selected_col_key_l = selected_col_keys.split(',')
        assert len(self.selected_col_key_l) == len(selected_col_ids.split(','))

        self.dataset = OFAFileDataset(
            file_path=file_path,
            selected_col_ids=selected_col_ids,
            dtypes=dtypes,
            separator=separator,
            cached_index=cached_index)
        self.preprocessor = OfaPreprocessor(model_dir, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value_l = self.dataset[index]
        data = dict()
        for key, value in zip(self.selected_col_key_l, value_l):
            data[key] = value
        return self.preprocessor(data)

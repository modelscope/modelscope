# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os

import datasets
from datasets import IterableDataset
from PIL import Image

from modelscope.utils.constant import EXTENSIONS_TO_LOAD
from modelscope.utils.logger import get_logger

logger = get_logger()


class ExternalDataset(object):

    def __init__(self, split_path_dict, config_kwargs):
        self.split_path_dict = split_path_dict
        self.config_kwargs = copy.deepcopy(config_kwargs)
        self.config_kwargs.update({'split_config': split_path_dict})
        self.ext_dataset = None
        self.split_data_files = {k: [] for k, _ in split_path_dict.items()}
        file_ext = ''

        for split_name, split_dir in split_path_dict.items():
            if isinstance(split_dir, str) and os.path.isdir(split_dir):
                split_file_names = os.listdir(split_dir)
                set_files_exts = set([
                    os.path.splitext(file_name)[-1].strip('.')
                    for file_name in split_file_names
                ])
                if '' in set_files_exts:
                    continue
                # ensure these files have same extensions
                if len(set_files_exts) != 1:
                    supported_exts = ','.join(EXTENSIONS_TO_LOAD.keys())
                    logger.error(
                        f'Split-{split_name} has been ignored, please flatten your folder structure, '
                        f'and make sure these files have same extensions. '
                        f'Supported extensions: {supported_exts} .')
                    continue
                file_ext = list(set_files_exts)[0]
                if file_ext not in EXTENSIONS_TO_LOAD:
                    continue

                split_file_paths = [
                    os.path.join(split_dir, file_name)
                    for file_name in split_file_names
                ]
                self.split_data_files[split_name] = split_file_paths

        if file_ext:
            file_ext = EXTENSIONS_TO_LOAD.get(file_ext)
            self.ext_dataset = datasets.load_dataset(
                file_ext, data_files=self.split_data_files, **config_kwargs)

    def __len__(self):
        return len(self.split_path_dict
                   ) if not self.ext_dataset else self.ext_dataset.__len__()

    def __getitem__(self, item):
        if not self.ext_dataset:
            return self.split_path_dict.get(item)
        else:
            return self.ext_dataset.__getitem__(item)

    def __iter__(self):
        if not self.ext_dataset:
            for k, v in self.split_path_dict.items():
                yield k, v
        else:
            for k, v in self.ext_dataset.items():
                yield k, v


class NativeIterableDataset(IterableDataset):
    """The modelscope iterable dataset class."""

    def __init__(self, ex_iterable, info, split):
        super().__init__(ex_iterable=ex_iterable, info=info, split=split)

    def __iter__(self):
        for key, entity in self._iter():
            if isinstance(entity, dict):
                ret = {}
                for k, v in entity.items():
                    ret[k] = v
                    if k.endswith(':FILE'):
                        dl_manager = self._ex_iterable.kwargs.get('dl_manager')
                        ex_cache_path = dl_manager.download_and_extract(v)
                        ret[k] = ex_cache_path
                        if k.endswith('Image:FILE'):
                            ret[k + ':Object'] = Image.open(fp=ex_cache_path)
                        if k.endswith('Audio:FILE'):
                            import torchaudio
                            waveform_and_rate = torchaudio.load(ex_cache_path)
                            ret[k + ':Object'] = waveform_and_rate
                entity = ret

            yield entity

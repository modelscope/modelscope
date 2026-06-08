# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
from itertools import islice

import datasets
from datasets import IterableDataset
from tqdm.auto import tqdm

from modelscope.utils.constant import EXTENSIONS_TO_LOAD
from modelscope.utils.logger import get_logger

logger = get_logger()


class ExternalDataset(object):
    """Dataset class for custom datasets."""

    def __init__(self, split_path_dict, config_kwargs):
        self.split_path_dict = split_path_dict
        self.config_kwargs = copy.deepcopy(config_kwargs)
        self.config_kwargs.update({'split_config': self.split_path_dict})
        # dataset for specific extensions
        self.spec_extension_dataset = None
        self.split_data_files = {
            k: []
            for k, _ in self.split_path_dict.items()
        }
        self.custom_map = {}

        # the extension of file
        file_ext = ''
        for split_name, split_dir in self.split_path_dict.items():
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
            self.spec_extension_dataset = datasets.load_dataset(
                file_ext, data_files=self.split_data_files, **config_kwargs)

    def __len__(self):
        return len(
            self.split_path_dict
        ) if not self.spec_extension_dataset else self.spec_extension_dataset.__len__(
        )

    def __getitem__(self, item):
        if not self.spec_extension_dataset:
            return self.split_path_dict.get(item)
        else:
            return self.spec_extension_dataset.__getitem__(item)

    def __iter__(self):
        if not self.spec_extension_dataset:
            for k, v in self.split_path_dict.items():
                yield k, v
        else:
            for k, v in self.spec_extension_dataset.items():
                yield k, v


class NativeIterableDataset(IterableDataset):
    """The modelscope iterable dataset class."""

    def __init__(self, ex_iterable, info, split, stream_batch_size=1):
        super().__init__(ex_iterable=ex_iterable, info=info, split=split)
        self.stream_batch_size = stream_batch_size

    def __iter__(self):
        for item in tqdm(
                self.iter(
                    batch_size=self.stream_batch_size, drop_last_batch=False),
                desc='Overall progress',
                total=self.n_shards,
                dynamic_ncols=True):
            ret = self._download_item(item)

            yield ret

    def __len__(self):
        return self.n_shards

    def __getitem__(self, index):
        """
        Returns the item at index `index` in the dataset. Slice indexing is supported.
        """
        if isinstance(index, int):
            start = index
            stop = index + 1
            step = None
        else:
            start = index.start
            stop = index.stop
            step = index.step

        if step is not None and step <= 0:
            raise ValueError('step must be positive')

        for item in tqdm(
                islice(
                    self.iter(batch_size=1, drop_last_batch=False), start,
                    stop, step),
                desc='Slicing progress',
                dynamic_ncols=True):
            ret = self._download_item(item)

            yield ret

    def _download_item(self, item):
        ret = {}
        if isinstance(item, dict):
            try:
                for k, v in item.items():
                    ret[k] = v
                    if k.endswith(':FILE'):
                        dl_manager = self._ex_iterable.kwargs.get('dl_manager')
                        ex_cache_path = dl_manager.download_and_extract(v)
                        if isinstance(ex_cache_path, str):
                            ex_cache_path = [ex_cache_path]
                        ret[k] = ex_cache_path
                        ret[k.strip(':FILE')] = v

            except Exception as e:
                logger.error(e)
                ret = item
        else:
            ret = item

        return ret

    def head(self, n=5):
        """
        Returns the first n rows of the dataset.

        Args:
            n (int): Number of rows to return.

        Returns:
            list: The list of results, e.g. [{'id': 'abc123', 'text': 'hello world'}, ...]
        """
        # return self._head(n=n)
        res = []
        if n <= 0:
            return res
        iter_num = 0
        for item in self.__iter__():
            if iter_num >= n:
                break
            res.append(item)
            iter_num += 1
        return res

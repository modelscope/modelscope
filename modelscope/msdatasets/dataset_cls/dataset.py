# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import math
import os
from itertools import islice

import datasets
import pandas as pd
from datasets import IterableDataset
from tqdm import tqdm

from modelscope.msdatasets.utils.maxcompute_utils import MaxComputeUtil
from modelscope.utils.constant import (DEFAULT_MAXCOMPUTE_ENDPOINT,
                                       EXTENSIONS_TO_LOAD, MaxComputeEnvs,
                                       VirgoDatasetConfig)
from modelscope.utils.logger import get_logger
from modelscope.utils.url_utils import fetch_csv_with_url, valid_url

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


class VirgoDataset(object):
    """Dataset class for Virgo.

    Attributes:
        _meta_content (str): Virgo meta data content, could be a url that contains csv file.
        _data_type (int): Virgo dataset type, 0-Standard virgo dataset; Others-User define dataset (to be supported)

    Examples:
        >>> from modelscope.msdatasets.dataset_cls.dataset import VirgoDataset
        >>> input_kwargs = {'metaContent': 'http://xxx-xxx/xxx.csv', 'samplingType': 0}
        >>> virgo_dataset = VirgoDataset(**input_kwargs)
        >>> print(virgo_dataset[1])
        >>> print(len(virgo_dataset))
        >>> for line in virgo_dataset:
        >>>     print(line)

        Note: If you set `download_virgo_files` to True by using
            MsDataset.load(dataset_name='your-virgo-dataset-id', hub=Hubs.virgo, download_virgo_files=True),
            you can get the cache file path of the virgo dataset, the column name is `cache_file`.
        >>> if virgo_dataset.download_virgo_files:
        >>>     print(virgo_dataset[1].get('cache_file'))
    """

    def __init__(self, **kwargs):

        self._meta_content: str = ''
        self.data_type: int = 0
        self.odps_table_name: str = ''
        self.odps_table_partition: str = None
        self._odps_utils: MaxComputeUtil = None
        self.config_kwargs = kwargs

        self._meta: pd.DataFrame = pd.DataFrame()

        self._meta_content = self.config_kwargs.pop(
            VirgoDatasetConfig.meta_content, '')
        self.data_type = self.config_kwargs.pop(
            VirgoDatasetConfig.sampling_type, 0)

        self._check_variables()
        self._parse_meta()

        self.meta_content_cache_file = ''
        self.virgo_cache_dir = ''
        self.download_virgo_files: bool = False

        self.odps_table_ins = None
        self.odps_reader_ins = None
        self.odps_batch_size = self.config_kwargs.pop('odps_batch_size', 100)
        self.odps_limit = self.config_kwargs.pop('odps_limit', None)
        self.odps_drop_last = self.config_kwargs.pop('odps_drop_last', False)
        if self._odps_utils:
            self.odps_table_ins, self.odps_reader_ins = self._odps_utils.get_table_reader_ins(
                self.odps_table_name, self.odps_table_partition)

    def __getitem__(self, index):
        if self.odps_reader_ins:
            return MaxComputeUtil.gen_reader_item(
                reader=self.odps_reader_ins,
                index=index,
                batch_size_in=self.odps_batch_size,
                limit_in=self.odps_limit,
                drop_last_in=self.odps_drop_last,
                partitions=self.odps_table_ins.table_schema.partitions,
                columns=self.odps_table_ins.table_schema.names)
        return self._meta.iloc[index].to_dict()

    def __len__(self):
        if isinstance(self._meta, dict):
            return self._meta.get('odpsCount', 0)
        return len(self._meta)

    def __iter__(self):
        if self.odps_reader_ins:
            odps_batch_data = MaxComputeUtil.gen_reader_batch(
                reader=self.odps_reader_ins,
                batch_size_in=self.odps_batch_size,
                limit_in=self.odps_limit,
                drop_last_in=self.odps_drop_last,
                partitions=self.odps_table_ins.table_schema.partitions,
                columns=self.odps_table_ins.table_schema.names)
            for batch in odps_batch_data:
                yield batch
        else:
            for _, row in self._meta.iterrows():
                yield row.to_dict()

    @property
    def meta(self) -> pd.DataFrame:
        """
        Virgo meta data. Contains columns: id, meta_info, analysis_result, external_info and
            cache_file (if download_virgo_files is True).
        """
        return self._meta

    def _parse_meta(self):
        # Fetch csv content
        if isinstance(self._meta_content, str) and valid_url(
                self._meta_content):
            meta_content_df = fetch_csv_with_url(self._meta_content)
            self._meta = meta_content_df
        elif isinstance(self._meta_content, dict):
            self._meta = self._meta_content
            self.odps_table_name = self._meta.get('odpsTableName', '')
            self.odps_table_partition = self._meta.get('odpsTablePartition',
                                                       None)
            self._odps_utils = self._get_odps_info()
        else:
            raise 'The meta content must be url or dict.'

    @staticmethod
    def _get_odps_info() -> MaxComputeUtil:
        """
        Get MaxComputeUtil instance.

        Args:
            None

        Returns:
            MaxComputeUtil instance.
        """
        access_id = os.environ.get(MaxComputeEnvs.ACCESS_ID, '')
        access_key = os.environ.get(MaxComputeEnvs.ACCESS_SECRET_KEY, '')
        proj_name = os.environ.get(MaxComputeEnvs.PROJECT_NAME, '')
        endpoint = os.environ.get(MaxComputeEnvs.ENDPOINT,
                                  DEFAULT_MAXCOMPUTE_ENDPOINT)

        if not access_id or not access_key or not proj_name:
            raise ValueError(
                f'Please set MaxCompute envs for Virgo: {MaxComputeEnvs.ACCESS_ID}, '
                f'{MaxComputeEnvs.ACCESS_SECRET_KEY}, {MaxComputeEnvs.PROJECT_NAME}, '
                f'{MaxComputeEnvs.ENDPOINT}(default: http://service-corp.odps.aliyun-inc.com/api)'
            )

        return MaxComputeUtil(access_id, access_key, proj_name, endpoint)

    def _check_variables(self):
        """Check member variables in this class.
            1. Condition-1: self._meta_content cannot be empty
            2. Condition-2: self._meta_content must be url when self._data_type is 0
        """
        if not self._meta_content:
            raise 'Them meta content cannot be empty.'
        if self.data_type not in [0, 1]:
            raise 'Supported samplingType should be 0 or 1, others are not supported yet.'
        if self.data_type == 0 and not valid_url(self._meta_content):
            raise 'The meta content must be url when data type is 0.'
        if self.data_type == 1 and not isinstance(self._meta_content, dict):
            raise 'The meta content must be dict when data type is 1.'

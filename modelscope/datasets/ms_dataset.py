import os
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Union)

import numpy as np
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from datasets.config import TF_AVAILABLE, TORCH_AVAILABLE
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES
from datasets.utils.file_utils import (is_relative_path,
                                       relative_to_absolute_path)

from modelscope.datasets.config import MS_DATASETS_CACHE
from modelscope.datasets.utils.ms_api import MsApi
from modelscope.utils.constant import Hubs
from modelscope.utils.logger import get_logger

logger = get_logger()


def format_list(para) -> List:
    if para is None:
        para = []
    elif isinstance(para, str):
        para = [para]
    elif len(set(para)) < len(para):
        raise ValueError(f'List columns contains duplicates: {para}')
    return para


class MsDataset:
    _hf_ds = None  # holds the underlying HuggingFace Dataset
    """A MsDataset backed by hugging face Dataset."""

    def __init__(self, hf_ds: Dataset, target: Optional[str] = None):
        self._hf_ds = hf_ds
        self.target = target

    def __iter__(self):
        for item in self._hf_ds:
            if self.target is not None:
                yield item[self.target]
            else:
                yield item

    def __getitem__(self, key):
        return self._hf_ds[key]

    @classmethod
    def from_hf_dataset(cls,
                        hf_ds: Dataset,
                        target: str = None) -> Union[dict, 'MsDataset']:
        if isinstance(hf_ds, Dataset):
            return cls(hf_ds, target)
        if len(hf_ds.keys()) == 1:
            return cls(next(iter(hf_ds.values())), target)
        return {k: cls(v, target) for k, v in hf_ds.items()}

    @staticmethod
    def load(
        dataset_name: Union[str, list],
        target: Optional[str] = None,
        version: Optional[str] = None,
        hub: Optional[Hubs] = Hubs.modelscope,
        subset_name: Optional[str] = None,
        split: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str],
                                   Mapping[str, Union[str,
                                                      Sequence[str]]]]] = None
    ) -> Union[dict, 'MsDataset']:
        """Load a MsDataset from the ModelScope Hub, Hugging Face Hub, urls, or a local dataset.
            Args:

                dataset_name (str): Path or name of the dataset.
                target (str, optional): Name of the column to output.
                version (str, optional): Version of the dataset script to load:
                subset_name (str, optional): Defining the subset_name of the dataset.
                data_dir (str, optional): Defining the data_dir of the dataset configuration. I
                data_files (str or Sequence or Mapping, optional): Path(s) to source data file(s).
                split (str, optional): Which split of the data to load.
                hub (Hubs, optional): When loading from a remote hub, where it is from

            Returns:
                MsDataset (obj:`MsDataset`): MsDataset object for a certain dataset.
            """
        if hub == Hubs.huggingface:
            dataset = hf_load_dataset(
                dataset_name,
                name=subset_name,
                revision=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files)
            return MsDataset.from_hf_dataset(dataset, target=target)
        else:
            return MsDataset._load_ms_dataset(
                dataset_name,
                target=target,
                subset_name=subset_name,
                version=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files)

    @staticmethod
    def _load_ms_dataset(
        dataset_name: Union[str, list],
        target: Optional[str] = None,
        version: Optional[str] = None,
        subset_name: Optional[str] = None,
        split: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str],
                                   Mapping[str, Union[str,
                                                      Sequence[str]]]]] = None
    ) -> Union[dict, 'MsDataset']:
        if isinstance(dataset_name, str):
            use_hf = False
            if dataset_name in _PACKAGED_DATASETS_MODULES or os.path.isdir(dataset_name) or \
                    (os.path.isfile(dataset_name) and dataset_name.endswith('.py')):
                use_hf = True
            elif is_relative_path(dataset_name):
                ms_api = MsApi()
                dataset_scripts = ms_api.fetch_dataset_scripts(
                    dataset_name, version)
                if 'py' in dataset_scripts:  # dataset copied from hf datasets
                    dataset_name = dataset_scripts['py'][0]
                    use_hf = True
            else:
                raise FileNotFoundError(
                    f"Couldn't find a dataset script at {relative_to_absolute_path(dataset_name)} "
                    f'or any data file in the same directory.')

            if use_hf:
                dataset = hf_load_dataset(
                    dataset_name,
                    name=subset_name,
                    revision=version,
                    split=split,
                    data_dir=data_dir,
                    data_files=data_files,
                    cache_dir=MS_DATASETS_CACHE)
            else:
                # TODO load from ms datahub
                raise NotImplementedError(
                    f'Dataset {dataset_name} load from modelscope datahub to be implemented in '
                    f'the future')
        elif isinstance(dataset_name, list):
            if target is None:
                target = 'target'
            dataset = Dataset.from_dict({target: dataset_name})
        else:
            raise TypeError('path must be a str or a list, but got'
                            f' {type(dataset_name)}')
        return MsDataset.from_hf_dataset(dataset, target=target)

    def to_torch_dataset_with_processors(
        self,
        preprocessors: Union[Callable, List[Callable]],
        columns: Union[str, List[str]] = None,
    ):
        preprocessor_list = preprocessors if isinstance(
            preprocessors, list) else [preprocessors]

        columns = format_list(columns)

        columns = [
            key for key in self._hf_ds.features.keys() if key in columns
        ]
        sample = next(iter(self._hf_ds))

        sample_res = {k: np.array(sample[k]) for k in columns}
        for processor in preprocessor_list:
            sample_res.update(
                {k: np.array(v)
                 for k, v in processor(sample).items()})

        def is_numpy_number(value):
            return np.issubdtype(value.dtype, np.integer) or np.issubdtype(
                value.dtype, np.floating)

        retained_columns = []
        for k in sample_res.keys():
            if not is_numpy_number(sample_res[k]):
                logger.warning(
                    f'Data of column {k} is non-numeric, will be removed')
                continue
            retained_columns.append(k)

        import torch

        class MsIterableDataset(torch.utils.data.IterableDataset):

            def __init__(self, dataset: Iterable):
                super(MsIterableDataset).__init__()
                self.dataset = dataset

            def __iter__(self):
                for item_dict in self.dataset:
                    res = {
                        k: np.array(item_dict[k])
                        for k in columns if k in retained_columns
                    }
                    for preprocessor in preprocessor_list:
                        res.update({
                            k: np.array(v)
                            for k, v in preprocessor(item_dict).items()
                            if k in retained_columns
                        })
                    yield res

        return MsIterableDataset(self._hf_ds)

    def to_torch_dataset(
        self,
        columns: Union[str, List[str]] = None,
        preprocessors: Union[Callable, List[Callable]] = None,
        **format_kwargs,
    ):
        """Create a torch.utils.data.Dataset from the MS Dataset. The torch.utils.data.Dataset can be passed to
           torch.utils.data.DataLoader.

        Args:
            preprocessors (Callable or List[Callable], default None): (list of) Preprocessor object used to process
                every sample of the dataset. The output type of processors is dict, and each numeric field of the dict
                will be used as a field of torch.utils.data.Dataset.
            columns (str or List[str], default None): Dataset column(s) to be loaded (numeric data only). If the
                preprocessor is None, the arg columns must have at least one column. If the `preprocessors` is not None,
                the output fields of processors will also be added.
            format_kwargs: A `dict` of arguments to be passed to the `torch.tensor`.

        Returns:
            :class:`tf.data.Dataset`

        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                'The function to_torch_dataset requires pytorch to be installed'
            )
        if preprocessors is not None:
            return self.to_torch_dataset_with_processors(preprocessors)
        else:
            self._hf_ds.reset_format()
            self._hf_ds.set_format(
                type='torch', columns=columns, format_kwargs=format_kwargs)
            return self._hf_ds

    def to_tf_dataset_with_processors(
        self,
        batch_size: int,
        shuffle: bool,
        preprocessors: Union[Callable, List[Callable]],
        drop_remainder: bool = None,
        prefetch: bool = True,
        label_cols: Union[str, List[str]] = None,
        columns: Union[str, List[str]] = None,
    ):
        preprocessor_list = preprocessors if isinstance(
            preprocessors, list) else [preprocessors]

        label_cols = format_list(label_cols)
        columns = format_list(columns)
        cols_to_retain = list(set(label_cols + columns))
        retained_columns = [
            key for key in self._hf_ds.features.keys() if key in cols_to_retain
        ]
        import tensorflow as tf
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            np.arange(len(self._hf_ds), dtype=np.int64))
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=len(self._hf_ds))

        def func(i, return_dict=False):
            i = int(i)
            res = {k: np.array(self._hf_ds[i][k]) for k in retained_columns}
            for preprocessor in preprocessor_list:
                # TODO preprocessor output may have the same key
                res.update({
                    k: np.array(v)
                    for k, v in preprocessor(self._hf_ds[i]).items()
                })
            if return_dict:
                return res
            return tuple(list(res.values()))

        sample_res = func(0, True)

        @tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
        def fetch_function(i):
            output = tf.numpy_function(
                func,
                inp=[i],
                Tout=[
                    tf.dtypes.as_dtype(val.dtype)
                    for val in sample_res.values()
                ],
            )
            return {key: output[i] for i, key in enumerate(sample_res)}

        tf_dataset = tf_dataset.map(
            fetch_function, num_parallel_calls=tf.data.AUTOTUNE)
        if label_cols:

            def split_features_and_labels(input_batch):
                labels = {
                    key: tensor
                    for key, tensor in input_batch.items() if key in label_cols
                }
                if len(input_batch) == 1:
                    input_batch = next(iter(input_batch.values()))
                if len(labels) == 1:
                    labels = next(iter(labels.values()))
                return input_batch, labels

            tf_dataset = tf_dataset.map(split_features_and_labels)

        elif len(columns) == 1:
            tf_dataset = tf_dataset.map(lambda x: next(iter(x.values())))
        if batch_size > 1:
            tf_dataset = tf_dataset.batch(
                batch_size, drop_remainder=drop_remainder)

        if prefetch:
            tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return tf_dataset

    def to_tf_dataset(
        self,
        batch_size: int,
        shuffle: bool,
        preprocessors: Union[Callable, List[Callable]] = None,
        columns: Union[str, List[str]] = None,
        collate_fn: Callable = None,
        drop_remainder: bool = None,
        collate_fn_args: Dict[str, Any] = None,
        label_cols: Union[str, List[str]] = None,
        prefetch: bool = True,
    ):
        """Create a tf.data.Dataset from the MS Dataset. This tf.data.Dataset can be passed to tf methods like
           model.fit() or model.predict().

        Args:
            batch_size (int): Number of samples in a single batch.
            shuffle(bool): Shuffle the dataset order.
            preprocessors (Callable or List[Callable], default None): (list of) Preprocessor object used to process
                every sample of the dataset. The output type of processors is dict, and each field of the dict will be
                used as a field of the tf.data. Dataset. If the `preprocessors` is None, the `collate_fn`
                shouldn't be None.
            columns (str or List[str], default None): Dataset column(s) to be loaded. If the preprocessor is None,
                the arg columns must have at least one column. If the `preprocessors` is not None, the output fields of
                processors will also be added.
            collate_fn(Callable, default None): A callable object used to collect lists of samples into a batch. If
                the `preprocessors` is None, the `collate_fn` shouldn't be None.
            drop_remainder(bool, default None): Drop the last incomplete batch when loading.
            collate_fn_args (Dict, optional): A `dict` of arguments to be passed to the`collate_fn`.
            label_cols (str or List[str], defalut None): Dataset column(s) to load as labels.
            prefetch (bool, default True): Prefetch data.

        Returns:
            :class:`tf.data.Dataset`

        """
        if not TF_AVAILABLE:
            raise ImportError(
                'The function to_tf_dataset requires Tensorflow to be installed.'
            )
        if preprocessors is not None:
            return self.to_tf_dataset_with_processors(
                batch_size,
                shuffle,
                preprocessors,
                drop_remainder=drop_remainder,
                prefetch=prefetch,
                label_cols=label_cols,
                columns=columns)

        if collate_fn is None:
            logger.error(
                'The `preprocessors` and the `collate_fn` should`t be both None.'
            )
            return None
        self._hf_ds.reset_format()
        return self._hf_ds.to_tf_dataset(
            columns,
            batch_size,
            shuffle,
            collate_fn,
            drop_remainder=drop_remainder,
            collate_fn_args=collate_fn_args,
            label_cols=label_cols,
            prefetch=prefetch)

    def to_hf_dataset(self) -> Dataset:
        self._hf_ds.reset_format()
        return self._hf_ds

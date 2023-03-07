# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import warnings
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Union)

import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES
from datasets.utils.file_utils import is_relative_path

from modelscope.hub.repository import DatasetRepository
from modelscope.msdatasets.context.dataset_context_config import \
    DatasetContextConfig
from modelscope.msdatasets.data_loader.data_loader_manager import (
    LocalDataLoaderManager, LocalDataLoaderType, RemoteDataLoaderManager,
    RemoteDataLoaderType)
from modelscope.msdatasets.dataset_cls.dataset import (ExternalDataset,
                                                       NativeIterableDataset)
from modelscope.msdatasets.task_datasets.builder import build_task_dataset
from modelscope.msdatasets.utils.delete_utils import DatasetDeleteManager
from modelscope.msdatasets.utils.upload_utils import DatasetUploadManager
from modelscope.utils.config import ConfigDict
from modelscope.utils.config_ds import MS_DATASETS_CACHE
from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       DEFAULT_DATASET_REVISION, DownloadMode,
                                       Hubs, UploadMode)
from modelscope.utils.import_utils import is_tf_available, is_torch_available
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
    """
    ModelScope Dataset (aka, MsDataset) is backed by a huggingface Dataset to
    provide efficient data access and local storage managements. On top of
    that, MsDataset supports the data integration and interactions with multiple
    remote hubs, particularly, ModelScope's own Dataset-hub. MsDataset also
    abstracts away data-access details with other remote storage, including both
    general external web-hosted data and cloud storage such as OSS.
    """
    # the underlying huggingface Dataset
    _hf_ds = None

    def __init__(self,
                 ds_instance: Union[Dataset, IterableDataset, ExternalDataset],
                 target: Optional[str] = None):
        self._hf_ds = ds_instance
        if target is not None and target not in self._hf_ds.features:
            raise TypeError(
                f'"target" must be a column of the dataset({list(self._hf_ds.features.keys())}, but got {target}'
            )
        self.target = target

    def __iter__(self):
        for item in self._hf_ds:
            if self.target is not None:
                yield item[self.target]
            else:
                yield item

    def __getitem__(self, key):
        return self._hf_ds[key]

    def __len__(self):
        if isinstance(self._hf_ds, IterableDataset) or isinstance(
                self._hf_ds, NativeIterableDataset):
            logger.error(
                f'object of type `{self._hf_ds.__class__.__name__}` has no __len__()'
            )
            return None
        return len(self._hf_ds)

    @property
    def ds_instance(self):
        return self._hf_ds

    @property
    def config_kwargs(self):
        if isinstance(self._hf_ds, ExternalDataset):
            return self._hf_ds.config_kwargs
        else:
            return None

    @classmethod
    def from_hf_dataset(cls,
                        hf_ds: Union[Dataset, DatasetDict, ExternalDataset],
                        target: str = None) -> Union[dict, 'MsDataset']:
        r"""
        @deprecated
        This method is deprecated and may be removed in future releases, please use `to_ms_dataset()` instead.
        """
        warnings.warn(
            'from_hf_dataset is deprecated, please use to_ms_dataset instead.',
            DeprecationWarning)
        if isinstance(hf_ds, Dataset):
            return cls(hf_ds, target)
        elif isinstance(hf_ds, DatasetDict):
            if len(hf_ds.keys()) == 1:
                return cls(next(iter(hf_ds.values())), target)
            return {k: cls(v, target) for k, v in hf_ds.items()}
        elif isinstance(hf_ds, ExternalDataset):
            return cls(hf_ds)
        else:
            raise TypeError(
                f'"hf_ds" must be a Dataset or DatasetDict, but got {type(hf_ds)}'
            )

    @classmethod
    def to_ms_dataset(cls,
                      ds_instance: Union[Dataset, DatasetDict, ExternalDataset,
                                         NativeIterableDataset,
                                         IterableDataset, IterableDatasetDict],
                      target: str = None) -> Union[dict, 'MsDataset']:
        """Convert input to `MsDataset` instance."""
        if isinstance(ds_instance, Dataset):
            return cls(ds_instance, target)
        elif isinstance(ds_instance, DatasetDict):
            if len(ds_instance.keys()) == 1:
                return cls(next(iter(ds_instance.values())), target)
            return {k: cls(v, target) for k, v in ds_instance.items()}
        elif isinstance(ds_instance, ExternalDataset):
            return cls(ds_instance)
        elif isinstance(ds_instance, NativeIterableDataset):
            return cls(ds_instance)
        elif isinstance(ds_instance, IterableDataset):
            return cls(ds_instance)
        elif isinstance(ds_instance, IterableDatasetDict):
            if len(ds_instance.keys()) == 1:
                return cls(next(iter(ds_instance.values())), target)
            return {k: cls(v, target) for k, v in ds_instance.items()}
        else:
            raise TypeError(
                f'"ds_instance" must be a Dataset or DatasetDict, but got {type(ds_instance)}'
            )

    @staticmethod
    def load(
        dataset_name: Union[str, list],
        namespace: Optional[str] = DEFAULT_DATASET_NAMESPACE,
        target: Optional[str] = None,
        version: Optional[str] = DEFAULT_DATASET_REVISION,
        hub: Optional[Hubs] = Hubs.modelscope,
        subset_name: Optional[str] = None,
        split: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str],
                                   Mapping[str, Union[str,
                                                      Sequence[str]]]]] = None,
        download_mode: Optional[DownloadMode] = DownloadMode.
        REUSE_DATASET_IF_EXISTS,
        cache_dir: Optional[str] = MS_DATASETS_CACHE,
        use_streaming: Optional[bool] = False,
        **config_kwargs,
    ) -> Union[dict, 'MsDataset', NativeIterableDataset]:
        """Load a MsDataset from the ModelScope Hub, Hugging Face Hub, urls, or a local dataset.

            Args:
                dataset_name (str): Path or name of the dataset.
                    The form of `namespace/dataset_name` is also supported.
                namespace(str, optional): Namespace of the dataset. It should not be None if you load a remote dataset
                    from Hubs.modelscope,
                namespace (str, optional):
                    Namespace of the dataset. It should not be None if you load a remote dataset
                    from Hubs.modelscope,
                target (str, optional): Name of the column to output.
                version (str, optional): Version of the dataset script to load:
                subset_name (str, optional): Defining the subset_name of the dataset.
                data_dir (str, optional): Defining the data_dir of the dataset configuration. I
                data_files (str or Sequence or Mapping, optional): Path(s) to source data file(s).
                split (str, optional): Which split of the data to load.
                hub (Hubs or str, optional): When loading from a remote hub, where it is from. default Hubs.modelscope
                download_mode (DownloadMode or str, optional):
                    How to treat existing datasets. default DownloadMode.REUSE_DATASET_IF_EXISTS
                config_kwargs (additional keyword arguments): Keyword arguments to be passed
                download_mode (DownloadMode or str, optional): How to treat existing datasets. default
                                                               DownloadMode.REUSE_DATASET_IF_EXISTS
                cache_dir (str, Optional): User-define local cache directory.
                use_streaming (bool, Optional): If set to True, no need to download all data files.
                                                Instead, it streams the data progressively, and returns
                                                NativeIterableDataset or a dict of NativeIterableDataset.
                **config_kwargs (additional keyword arguments): Keyword arguments to be passed

            Returns:
                MsDataset (MsDataset): MsDataset object for a certain dataset.
            """

        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        hub = Hubs(hub or Hubs.modelscope)

        if not isinstance(dataset_name, str) and not isinstance(
                dataset_name, list):
            raise TypeError(
                f'dataset_name must be `str` or `list`, but got {type(dataset_name)}'
            )

        if isinstance(dataset_name, list):
            if target is None:
                target = 'target'
            dataset_inst = Dataset.from_dict({target: dataset_name})
            return MsDataset.to_ms_dataset(dataset_inst, target=target)

        dataset_name = os.path.expanduser(dataset_name)
        is_local_path = os.path.exists(dataset_name)
        if is_relative_path(dataset_name) and dataset_name.count(
                '/') == 1 and not is_local_path:
            dataset_name_split = dataset_name.split('/')
            namespace = dataset_name_split[0].strip()
            dataset_name = dataset_name_split[1].strip()
            if not namespace or not dataset_name:
                raise 'The dataset_name should be in the form of `namespace/dataset_name` or `dataset_name`.'

        # Init context config
        dataset_context_config = DatasetContextConfig(
            dataset_name=dataset_name,
            namespace=namespace,
            version=version,
            subset_name=subset_name,
            split=split,
            target=target,
            hub=hub,
            data_dir=data_dir,
            data_files=data_files,
            download_mode=download_mode,
            cache_root_dir=cache_dir,
            use_streaming=use_streaming,
            **config_kwargs)

        # Load from local disk
        if dataset_name in _PACKAGED_DATASETS_MODULES or os.path.isdir(
                dataset_name) or os.path.isfile(dataset_name):
            dataset_inst = LocalDataLoaderManager(
                dataset_context_config).load_dataset(
                    LocalDataLoaderType.HF_DATA_LOADER)
            return MsDataset.to_ms_dataset(dataset_inst, target=target)
        # Load from the huggingface hub
        elif hub == Hubs.huggingface:
            dataset_inst = RemoteDataLoaderManager(
                dataset_context_config).load_dataset(
                    RemoteDataLoaderType.HF_DATA_LOADER)
            return MsDataset.to_ms_dataset(dataset_inst, target=target)
        # Load from the modelscope hub
        elif hub == Hubs.modelscope:
            dataset_inst = RemoteDataLoaderManager(
                dataset_context_config).load_dataset(
                    RemoteDataLoaderType.MS_DATA_LOADER)
            return MsDataset.to_ms_dataset(dataset_inst, target=target)
        else:
            raise 'Please adjust input args to specify a loading mode, we support following scenes: ' \
                  'loading from local disk, huggingface hub and modelscope hub.'

    def to_torch_dataset_with_processors(
        self,
        preprocessors: Union[Callable, List[Callable]],
        columns: Union[str, List[str]] = None,
        to_tensor: bool = True,
    ):
        import torch
        preprocessor_list = preprocessors if isinstance(
            preprocessors, list) else [preprocessors]

        columns = format_list(columns)

        columns = [
            key for key in self._hf_ds.features.keys() if key in columns
        ]
        retained_columns = []
        if to_tensor:
            sample = next(iter(self._hf_ds))

            sample_res = {k: np.array(sample[k]) for k in columns}
            for processor in preprocessor_list:
                sample_res.update(
                    {k: np.array(v)
                     for k, v in processor(sample).items()})

            def is_numpy_number(value):
                return np.issubdtype(value.dtype, np.integer) or np.issubdtype(
                    value.dtype, np.floating)

            for k in sample_res.keys():
                if not is_numpy_number(sample_res[k]):
                    logger.warning(
                        f'Data of column {k} is non-numeric, will be removed')
                    continue
                retained_columns.append(k)

        class MsMapDataset(torch.utils.data.Dataset):

            def __init__(self, dataset: Iterable, preprocessor_list,
                         retained_columns, columns, to_tensor):
                super(MsDataset).__init__()
                self.dataset = dataset
                self.preprocessor_list = preprocessor_list
                self.to_tensor = to_tensor
                self.retained_columns = retained_columns
                self.columns = columns

            def __len__(self):
                return len(self.dataset)

            def type_converter(self, x):
                import torch
                if self.to_tensor:
                    return torch.tensor(x)
                else:
                    return x

            def __getitem__(self, index):
                item_dict = self.dataset[index]
                res = {
                    k: self.type_converter(item_dict[k])
                    for k in self.columns
                    if (not self.to_tensor) or k in self.retained_columns
                }
                for preprocessor in self.preprocessor_list:
                    res.update({
                        k: self.type_converter(v)
                        for k, v in preprocessor(item_dict).items()
                        if (not self.to_tensor) or k in self.retained_columns
                    })
                return res

        return MsMapDataset(self._hf_ds, preprocessor_list, retained_columns,
                            columns, to_tensor)

    def to_torch_dataset(
        self,
        columns: Union[str, List[str]] = None,
        preprocessors: Union[Callable, List[Callable]] = None,
        task_name: str = None,
        task_data_config: ConfigDict = None,
        to_tensor: bool = True,
        **format_kwargs,
    ):
        """Create a torch.utils.data.Dataset from the MS Dataset. The torch.utils.data.Dataset can be passed to
           torch.utils.data.DataLoader.

        Args:
            preprocessors (Callable or List[Callable], default None): (list of) Preprocessor object used to process
                every sample of the dataset. The output type of processors is dict, and each (numeric) field of the dict
                will be used as a field of torch.utils.data.Dataset.
            columns (str or List[str], default None): Dataset column(s) to be loaded (numeric data only if
                `to_tensor` is True). If the preprocessor is None, the arg columns must have at least one column.
                If the `preprocessors` is not None, the output fields of processors will also be added.
            task_name (str, default None):  task name, refer to :obj:`Tasks` for more details
            task_data_config (ConfigDict, default None): config dict for model object.
            to_tensor (bool, default None): whether convert the data types of dataset column(s) to torch.tensor or not.
            format_kwargs: A `dict` of arguments to be passed to the `torch.tensor`.

        Returns:
            :class:`tf.data.Dataset`

        """
        if not is_torch_available():
            raise ImportError(
                'The function to_torch_dataset requires pytorch to be installed'
            )
        if isinstance(self._hf_ds, ExternalDataset):
            task_data_config.update({'preprocessor': preprocessors})
            task_data_config.update(self._hf_ds.config_kwargs)
            return build_task_dataset(task_data_config, task_name)
        if preprocessors is not None:
            return self.to_torch_dataset_with_processors(
                preprocessors, columns=columns, to_tensor=to_tensor)
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

        from tensorflow.data.experimental import AUTOTUNE
        tf_dataset = tf_dataset.map(
            fetch_function, num_parallel_calls=AUTOTUNE)
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
            tf_dataset = tf_dataset.prefetch(AUTOTUNE)
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
        if not is_tf_available():
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

    def remap_columns(self, column_mapping: Dict[str, str]) -> Dataset:
        """
        Rename columns and return the underlying hf dataset directly
        TODO: support native MsDataset column rename.
        Args:
            column_mapping: the mapping of the original and new column names
        Returns:
            underlying hf dataset
        """
        self._hf_ds.reset_format()
        return self._hf_ds.rename_columns(column_mapping)

    @staticmethod
    def upload(
            object_name: str,
            local_file_path: str,
            dataset_name: str,
            namespace: Optional[str] = DEFAULT_DATASET_NAMESPACE,
            version: Optional[str] = DEFAULT_DATASET_REVISION,
            num_processes: Optional[int] = None,
            chunksize: Optional[int] = 1,
            filter_hidden_files: Optional[bool] = True,
            upload_mode: Optional[UploadMode] = UploadMode.OVERWRITE) -> None:
        """Upload dataset file or directory to the ModelScope Hub. Please log in to the ModelScope Hub first.

        Args:
            object_name (str): The object name on ModelScope, in the form of your-dataset-name.zip or your-dataset-name
            local_file_path (str): Local file or directory to upload
            dataset_name (str): Name of the dataset
            namespace(str, optional): Namespace of the dataset
            version: Optional[str]: Version of the dataset
            num_processes: Optional[int]: The number of processes used for multiprocess uploading.
                This is only applicable when local_file_path is a directory, and we are uploading mutliple-files
                insided the directory. When None provided, the number returned by os.cpu_count() is used as default.
            chunksize: Optional[int]: The chunksize of objects to upload.
                For very long iterables using a large value for chunksize can make the job complete much faster than
                using the default value of 1. Available if local_file_path is a directory.
            filter_hidden_files: Optional[bool]: Whether to filter hidden files.
                Available if local_file_path is a directory.
            upload_mode: Optional[UploadMode]: How to upload objects from local.  Default: UploadMode.OVERWRITE, upload
                all objects from local, existing remote objects may be overwritten.

        Returns:
            None

        """
        if not object_name:
            raise ValueError('object_name cannot be empty!')

        _upload_manager = DatasetUploadManager(
            dataset_name=dataset_name, namespace=namespace, version=version)

        upload_mode = UploadMode(upload_mode or UploadMode.OVERWRITE)

        if os.path.isfile(local_file_path):
            _upload_manager.upload(
                object_name=object_name,
                local_file_path=local_file_path,
                upload_mode=upload_mode)
        elif os.path.isdir(local_file_path):
            _upload_manager.upload_dir(
                object_dir_name=object_name,
                local_dir_path=local_file_path,
                num_processes=num_processes,
                chunksize=chunksize,
                filter_hidden_files=filter_hidden_files,
                upload_mode=upload_mode)
        else:
            raise ValueError(
                f'{local_file_path} is not a valid file path or directory')

    @staticmethod
    def clone_meta(dataset_work_dir: str,
                   dataset_id: str,
                   revision: Optional[str] = DEFAULT_DATASET_REVISION,
                   auth_token: Optional[str] = None,
                   git_path: Optional[str] = None) -> None:
        """Clone meta-file of dataset from the ModelScope Hub.

        Args:
            dataset_work_dir (str): Current git working directory.
            dataset_id (str): Dataset id, in the form of your-namespace/your-dataset-name .
            revision (str, optional):
                revision of the model you want to clone from. Can be any of a branch, tag or commit hash
            auth_token (str, optional):
                token obtained when calling `HubApi.login()`. Usually you can safely ignore the parameter
                as the token is already saved when you login the first time, if None, we will use saved token.
            git_path (str, optional):
                The git command line path, if None, we use 'git'
        Returns:
            None
        """

        _repo = DatasetRepository(
            repo_work_dir=dataset_work_dir,
            dataset_id=dataset_id,
            revision=revision,
            auth_token=auth_token,
            git_path=git_path)
        clone_work_dir = _repo.clone()
        if clone_work_dir:
            logger.info('Already cloned repo to: {}'.format(clone_work_dir))
        else:
            logger.warning(
                'Repo dir already exists: {}'.format(clone_work_dir))

    @staticmethod
    def upload_meta(dataset_work_dir: str,
                    commit_message: str,
                    revision: Optional[str] = DEFAULT_DATASET_REVISION,
                    auth_token: Optional[str] = None,
                    git_path: Optional[str] = None,
                    force: bool = False) -> None:
        """Upload meta-file of dataset to the ModelScope Hub. Please clone the meta-data from the ModelScope Hub first.

        Args:
            dataset_work_dir (str): Current working directory.
            commit_message (str): Commit message.
            revision(`Optional[str]`):
                revision of the model you want to clone from. Can be any of a branch, tag or commit hash
            auth_token(`Optional[str]`):
                token obtained when calling `HubApi.login()`. Usually you can safely ignore the parameter
                as the token is already saved when you log in the first time, if None, we will use saved token.
            git_path:(`Optional[str]`):
                The git command line path, if None, we use 'git'
            force (Optional[bool]): whether to use forced-push.

        Returns:
            None

        """
        _repo = DatasetRepository(
            repo_work_dir=dataset_work_dir,
            dataset_id='',
            revision=revision,
            auth_token=auth_token,
            git_path=git_path)
        _repo.push(commit_message=commit_message, branch=revision, force=force)

    @staticmethod
    def delete(object_name: str,
               dataset_name: str,
               namespace: Optional[str] = DEFAULT_DATASET_NAMESPACE,
               version: Optional[str] = DEFAULT_DATASET_REVISION) -> str:
        """ Delete object of dataset. Please log in first and make sure you have permission to manage the dataset.

        Args:
            object_name (str): The object name of dataset to be deleted. Could be a name of file or directory. If it's
                directory, then ends with `/`.
                For example: your-data-name.zip, train/001/img_001.png, train/, ...
            dataset_name (str): Path or name of the dataset.
            namespace(str, optional): Namespace of the dataset.
            version (str, optional): Version of the dataset.

        Returns:
            res_msg (str): Response message.

        """
        _delete_manager = DatasetDeleteManager(
            dataset_name=dataset_name, namespace=namespace, version=version)
        resp_msg = _delete_manager.delete(object_name=object_name)
        logger.info(f'Object {object_name} successfully removed!')
        return resp_msg

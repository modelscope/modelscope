import os
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Union)

import json
import numpy as np
from datasets import Dataset, DatasetDict
from datasets import load_dataset as hf_load_dataset
from datasets.config import TF_AVAILABLE, TORCH_AVAILABLE
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES
from datasets.utils.download_manager import DownloadConfig
from datasets.utils.file_utils import (is_relative_path,
                                       relative_to_absolute_path)

from modelscope.hub.repository import DatasetRepository
from modelscope.utils.config import ConfigDict
from modelscope.utils.config_ds import MS_DATASETS_CACHE
from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       DEFAULT_DATASET_REVISION,
                                       DatasetFormations, DownloadMode, Hubs)
from modelscope.utils.logger import get_logger
from .task_datasets.builder import build_task_dataset
from .utils.dataset_builder import ExternalDataset
from .utils.dataset_utils import (get_dataset_files,
                                  get_target_dataset_structure,
                                  load_dataset_builder)
from .utils.download_utils import DatasetDownloadManager
from .utils.upload_utils import DatasetUploadManager

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

    def __init__(self, hf_ds: Dataset, target: Optional[str] = None):
        self._hf_ds = hf_ds
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
        return len(self._hf_ds)

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
        **config_kwargs,
    ) -> Union[dict, 'MsDataset']:
        """Load a MsDataset from the ModelScope Hub, Hugging Face Hub, urls, or a local dataset.
            Args:

                dataset_name (str): Path or name of the dataset.
                namespace(str, optional): Namespace of the dataset. It should not be None if you load a remote dataset
                from Hubs.modelscope,
                target (str, optional): Name of the column to output.
                version (str, optional): Version of the dataset script to load:
                subset_name (str, optional): Defining the subset_name of the dataset.
                data_dir (str, optional): Defining the data_dir of the dataset configuration. I
                data_files (str or Sequence or Mapping, optional): Path(s) to source data file(s).
                split (str, optional): Which split of the data to load.
                hub (Hubs or str, optional): When loading from a remote hub, where it is from. default Hubs.modelscope
                download_mode (DownloadMode or str, optional): How to treat existing datasets. default
                                                               DownloadMode.REUSE_DATASET_IF_EXISTS
                **config_kwargs (additional keyword arguments): Keyword arguments to be passed

            Returns:
                MsDataset (obj:`MsDataset`): MsDataset object for a certain dataset.
            """
        download_mode = DownloadMode(download_mode
                                     or DownloadMode.REUSE_DATASET_IF_EXISTS)
        hub = Hubs(hub or Hubs.modelscope)
        if hub == Hubs.huggingface:
            dataset = hf_load_dataset(
                dataset_name,
                name=subset_name,
                revision=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files,
                download_mode=download_mode.value,
                **config_kwargs)
            return MsDataset.from_hf_dataset(dataset, target=target)
        elif hub == Hubs.modelscope:
            return MsDataset._load_ms_dataset(
                dataset_name,
                namespace=namespace,
                target=target,
                subset_name=subset_name,
                version=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files,
                download_mode=download_mode,
                **config_kwargs)

    @staticmethod
    def _load_ms_dataset(dataset_name: Union[str, list],
                         namespace: Optional[str] = None,
                         target: Optional[str] = None,
                         version: Optional[str] = DEFAULT_DATASET_REVISION,
                         subset_name: Optional[str] = None,
                         split: Optional[str] = None,
                         data_dir: Optional[str] = None,
                         data_files: Optional[Union[
                             str, Sequence[str],
                             Mapping[str, Union[str, Sequence[str]]]]] = None,
                         download_mode: Optional[DownloadMode] = None,
                         **config_kwargs) -> Union[dict, 'MsDataset']:
        from modelscope.hub.api import HubApi
        api = HubApi()
        download_dataset = ''
        if isinstance(dataset_name, str):
            download_dataset = dataset_name
            dataset_formation = DatasetFormations.native
            if dataset_name in _PACKAGED_DATASETS_MODULES or os.path.isdir(dataset_name) or \
                    (os.path.isfile(dataset_name) and dataset_name.endswith('.py')):
                dataset_formation = DatasetFormations.hf_compatible
            elif is_relative_path(dataset_name) and dataset_name.count(
                    '/') == 0:
                dataset_scripts, dataset_formation, download_dir = api.fetch_dataset_scripts(
                    dataset_name, namespace, download_mode, version)
                # dataset organized to be compatible with hf format
                if dataset_formation == DatasetFormations.hf_compatible:
                    dataset_name = dataset_scripts['.py'][0]
            else:
                raise FileNotFoundError(
                    f"Couldn't find a dataset script at {relative_to_absolute_path(dataset_name)} "
                    f'or any data file in the same directory.')

            if dataset_formation == DatasetFormations.hf_compatible:
                dataset = hf_load_dataset(
                    dataset_name,
                    name=subset_name,
                    revision=version,
                    split=split,
                    data_dir=data_dir,
                    data_files=data_files,
                    cache_dir=MS_DATASETS_CACHE,
                    download_mode=download_mode.value,
                    **config_kwargs)
            else:
                dataset = MsDataset._load_from_ms(
                    dataset_name,
                    dataset_scripts,
                    download_dir,
                    namespace=namespace,
                    version=version,
                    subset_name=subset_name,
                    split=split,
                    download_mode=download_mode,
                    **config_kwargs)
        elif isinstance(dataset_name, list):
            if target is None:
                target = 'target'
            dataset = Dataset.from_dict({target: dataset_name})
        else:
            raise TypeError('path must be a str or a list, but got'
                            f' {type(dataset_name)}')

        if download_dataset:
            api.on_dataset_download(
                dataset_name=download_dataset, namespace=namespace)

        return MsDataset.from_hf_dataset(dataset, target=target)

    @staticmethod
    def _load_from_ms(dataset_name: str,
                      dataset_files: dict,
                      download_dir: str,
                      namespace: Optional[str] = None,
                      version: Optional[str] = DEFAULT_DATASET_REVISION,
                      subset_name: Optional[str] = None,
                      split: Optional[str] = None,
                      download_mode: Optional[DownloadMode] = None,
                      **config_kwargs) -> Union[Dataset, DatasetDict]:
        for json_path in dataset_files['.json']:
            if json_path.endswith(f'{dataset_name}.json'):
                with open(json_path, encoding='utf-8') as dataset_json_file:
                    dataset_json = json.load(dataset_json_file)
                break
        target_subset_name, target_dataset_structure = get_target_dataset_structure(
            dataset_json, subset_name, split)
        meta_map, file_map, args_map = get_dataset_files(
            target_dataset_structure, dataset_name, namespace, version)
        builder = load_dataset_builder(
            dataset_name,
            subset_name,
            namespace,
            meta_data_files=meta_map,
            zip_data_files=file_map,
            args_map=args_map,
            cache_dir=MS_DATASETS_CACHE,
            version=version,
            split=list(target_dataset_structure.keys()),
            **config_kwargs)

        download_config = DownloadConfig(
            cache_dir=download_dir,
            force_download=bool(
                download_mode == DownloadMode.FORCE_REDOWNLOAD),
            force_extract=bool(download_mode == DownloadMode.FORCE_REDOWNLOAD),
            use_etag=False,
        )

        dl_manager = DatasetDownloadManager(
            dataset_name=dataset_name,
            namespace=namespace,
            version=version,
            download_config=download_config,
            data_dir=download_dir,
        )
        builder.download_and_prepare(
            dl_manager=dl_manager,
            download_mode=download_mode.value,
            try_from_hf_gcs=False)

        ds = builder.as_dataset()
        return ds

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

        import math
        import torch

        class MsIterableDataset(torch.utils.data.IterableDataset):

            def __init__(self, dataset: Iterable):
                super(MsIterableDataset).__init__()
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:  # single-process data loading
                    iter_start = 0
                    iter_end = len(self.dataset)
                else:  # in a worker process
                    per_worker = math.ceil(
                        len(self.dataset) / float(worker_info.num_workers))
                    worker_id = worker_info.id
                    iter_start = worker_id * per_worker
                    iter_end = min(iter_start + per_worker, len(self.dataset))

                for idx in range(iter_start, iter_end):
                    item_dict = self.dataset[idx]
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
        task_name: str = None,
        task_data_config: ConfigDict = None,
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
            task_name (str, default None):  task name, refer to :obj:`Tasks` for more details
            task_data_config (ConfigDict, default None): config dict for model object.
            format_kwargs: A `dict` of arguments to be passed to the `torch.tensor`.

        Returns:
            :class:`tf.data.Dataset`

        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                'The function to_torch_dataset requires pytorch to be installed'
            )
        if isinstance(self._hf_ds, ExternalDataset):
            task_data_config.update({'preprocessor': preprocessors})
            return build_task_dataset(task_data_config, task_name,
                                      self._hf_ds.config_kwargs)
        if preprocessors is not None:
            return self.to_torch_dataset_with_processors(
                preprocessors, columns=columns)
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

    @staticmethod
    def upload(object_name: str,
               local_file_path: str,
               dataset_name: str,
               namespace: Optional[str] = DEFAULT_DATASET_NAMESPACE,
               version: Optional[str] = DEFAULT_DATASET_REVISION) -> None:
        """Upload dataset file to the ModelScope Hub. Please login to the ModelScope Hub first.

        Args:
            object_name (str): The object name on ModelScope, in the form of your-dataset-name.zip
            local_file_path (str): Local file to upload
            dataset_name (str): Name of the dataset
            namespace(str, optional): Namespace of the dataset
            version: Optional[str]: Version of the dataset

        Returns:
            None

        """
        from modelscope.hub.api import HubApi
        _hub_api = HubApi()
        cookies = _hub_api.check_cookies_upload_data(use_cookies=True)
        _upload_manager = DatasetUploadManager(
            dataset_name=dataset_name,
            namespace=namespace,
            version=version,
            cookies=cookies)
        _upload_manager.upload(object_name, local_file_path)

    @staticmethod
    def clone_meta(dataset_work_dir: str,
                   dataset_id: str,
                   revision: Optional[str] = DEFAULT_DATASET_REVISION,
                   auth_token: Optional[str] = None,
                   git_path: Optional[str] = None) -> None:
        """Clone meta-file of dataset from the ModelScope Hub.
        Args:
            dataset_work_dir (str): Current git working directory.
            dataset_id (str): Dataset id, It should be like your-namespace/your-dataset-name .
            revision(`Optional[str]`):
                revision of the model you want to clone from. Can be any of a branch, tag or commit hash
            auth_token(`Optional[str]`):
                token obtained when calling `HubApi.login()`. Usually you can safely ignore the parameter
                as the token is already saved when you login the first time, if None, we will use saved token.
            git_path:(`Optional[str]`):
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
            logger.warning('The repo working dir is already ex.')

    @staticmethod
    def upload_meta(dataset_work_dir: str,
                    dataset_id: str,
                    commit_message: str,
                    revision: Optional[str] = DEFAULT_DATASET_REVISION,
                    auth_token: Optional[str] = None,
                    git_path: Optional[str] = None,
                    force: bool = False) -> None:
        """Upload meta-file of dataset to the ModelScope Hub. Please clone the meta-data from the ModelScope Hub first.

        Args:
            dataset_work_dir (str): Current working directory.
            dataset_id (str): Dataset id, It should be like your-namespace/your-dataset-name .
            commit_message (str): Commit message.
            revision(`Optional[str]`):
                revision of the model you want to clone from. Can be any of a branch, tag or commit hash
            auth_token(`Optional[str]`):
                token obtained when calling `HubApi.login()`. Usually you can safely ignore the parameter
                as the token is already saved when you login the first time, if None, we will use saved token.
            git_path:(`Optional[str]`):
                The git command line path, if None, we use 'git'
            force (Optional[bool]): whether to use forced-push.

        Returns:
            None

        """
        _repo = DatasetRepository(
            repo_work_dir=dataset_work_dir,
            dataset_id=dataset_id,
            revision=revision,
            auth_token=auth_token,
            git_path=git_path)
        _repo.push(commit_message=commit_message, branch=revision, force=force)

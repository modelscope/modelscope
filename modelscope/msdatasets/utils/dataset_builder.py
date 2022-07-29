import os
from typing import Mapping, Sequence, Union

import datasets
import pandas as pd
import pyarrow as pa
from datasets.info import DatasetInfo
from datasets.packaged_modules import csv
from datasets.utils.filelock import FileLock

from modelscope.utils.logger import get_logger

logger = get_logger()


class MsCsvDatasetBuilder(csv.Csv):

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str,
        namespace: str,
        subset_name: str,
        hash: str,
        meta_data_files: Mapping[str, Union[str, Sequence[str]]],
        zip_data_files: Mapping[str, Union[str, Sequence[str]]] = None,
        **config_kwargs,
    ):
        super().__init__(
            cache_dir=cache_dir,
            name=subset_name,
            hash=hash,
            namespace=namespace,
            data_files=meta_data_files,
            **config_kwargs)

        self.name = dataset_name
        self.info.builder_name = self.name
        self._cache_dir = self._build_cache_dir()
        lock_path = os.path.join(
            self._cache_dir_root,
            self._cache_dir.replace(os.sep, '_') + '.lock')
        with FileLock(lock_path):
            # check if data exist
            if os.path.exists(self._cache_dir):
                if len(os.listdir(self._cache_dir)) > 0:
                    logger.info(
                        f'Overwrite dataset info from restored data version, cache_dir is {self._cache_dir}'
                    )
                    self.info = DatasetInfo.from_directory(self._cache_dir)
                # dir exists but no data, remove the empty dir as data aren't available anymore
                else:
                    logger.warning(
                        f'Old caching folder {self._cache_dir} for dataset {self.name} exists '
                        f'but not data were found. Removing it. ')
                    os.rmdir(self._cache_dir)
        self.zip_data_files = zip_data_files

    def _build_cache_dir(self):
        builder_data_dir = os.path.join(
            self._cache_dir_root,
            self._relative_data_dir(with_version=False, with_hash=True))

        return builder_data_dir

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(
                'At least one data file must be specified, but got none.')
        data_files = dl_manager.download_and_extract(self.config.data_files)
        zip_data_files = dl_manager.download_and_extract(self.zip_data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        'files': dl_manager.iter_files(files),
                        'base_dir': zip_data_files.get(split_name)
                    }))
        return splits

    def _generate_tables(self, files, base_dir):
        schema = pa.schema(self.config.features.type
                           ) if self.config.features is not None else None
        dtype = {
            name: dtype.to_pandas_dtype()
            for name, dtype in zip(schema.names, schema.types)
        } if schema else None
        for file_idx, file in enumerate(files):
            csv_file_reader = pd.read_csv(
                file,
                iterator=True,
                dtype=dtype,
                **self.config.read_csv_kwargs)
            transform_fields = []
            for field_name in csv_file_reader._engine.names:
                if field_name.endswith(':FILE'):
                    transform_fields.append(field_name)
            try:
                for batch_idx, df in enumerate(csv_file_reader):
                    for field_name in transform_fields:
                        if base_dir:
                            df[field_name] = df[field_name].apply(
                                lambda x: os.path.join(base_dir, x))
                    pa_table = pa.Table.from_pandas(df, schema=schema)
                    yield (file_idx, batch_idx), pa_table
            except ValueError as e:
                logger.error(
                    f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise

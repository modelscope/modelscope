import logging
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Union)

from datasets import Dataset, load_dataset

from maas_lib.utils.logger import get_logger

logger = get_logger()


class PyDataset:
    _hf_ds = None  # holds the underlying HuggingFace Dataset
    """A PyDataset backed by hugging face Dataset."""

    def __init__(self, hf_ds: Dataset):
        self._hf_ds = hf_ds
        self.target = None

    def __iter__(self):
        if isinstance(self._hf_ds, Dataset):
            for item in self._hf_ds:
                if self.target is not None:
                    yield item[self.target]
                else:
                    yield item
        else:
            for ds in self._hf_ds.values():
                for item in ds:
                    if self.target is not None:
                        yield item[self.target]
                    else:
                        yield item

    @classmethod
    def from_hf_dataset(cls,
                        hf_ds: Dataset,
                        target: str = None) -> 'PyDataset':
        dataset = cls(hf_ds)
        dataset.target = target
        return dataset

    @staticmethod
    def load(
        path: Union[str, list],
        target: Optional[str] = None,
        version: Optional[str] = None,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str],
                                   Mapping[str, Union[str,
                                                      Sequence[str]]]]] = None
    ) -> 'PyDataset':
        """Load a PyDataset from the MaaS Hub, Hugging Face Hub, urls, or a local dataset.
            Args:

                path (str): Path or name of the dataset.
                target (str, optional): Name of the column to output.
                version (str, optional): Version of the dataset script to load:
                name (str, optional): Defining the subset_name of the dataset.
                data_dir (str, optional): Defining the data_dir of the dataset configuration. I
                data_files (str or Sequence or Mapping, optional): Path(s) to source data file(s).
                split (str, optional): Which split of the data to load.

            Returns:
                PyDataset (obj:`PyDataset`): PyDataset object for a certain dataset.
            """
        if isinstance(path, str):
            dataset = load_dataset(
                path,
                name=name,
                revision=version,
                split=split,
                data_dir=data_dir,
                data_files=data_files)
        elif isinstance(path, list):
            if target is None:
                target = 'target'
            dataset = Dataset.from_dict({target: [p] for p in path})
        else:
            raise TypeError('path must be a str or a list, but got'
                            f' {type(path)}')
        return PyDataset.from_hf_dataset(dataset, target=target)

    def to_torch_dataset(
        self,
        columns: Union[str, List[str]] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        self._hf_ds.reset_format()
        self._hf_ds.set_format(
            type='torch',
            columns=columns,
            output_all_columns=output_all_columns,
            format_kwargs=format_kwargs)
        return self._hf_ds

    def to_tf_dataset(
        self,
        columns: Union[str, List[str]],
        batch_size: int,
        shuffle: bool,
        collate_fn: Callable,
        drop_remainder: bool = None,
        collate_fn_args: Dict[str, Any] = None,
        label_cols: Union[str, List[str]] = None,
        dummy_labels: bool = False,
        prefetch: bool = True,
    ):
        self._hf_ds.reset_format()
        return self._hf_ds.to_tf_dataset(
            columns,
            batch_size,
            shuffle,
            collate_fn,
            drop_remainder=drop_remainder,
            collate_fn_args=collate_fn_args,
            label_cols=label_cols,
            dummy_labels=dummy_labels,
            prefetch=prefetch)

    def to_hf_dataset(self) -> Dataset:
        self._hf_ds.reset_format()
        return self._hf_ds

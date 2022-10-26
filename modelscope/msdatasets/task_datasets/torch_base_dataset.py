# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, List, Tuple, Union

from torch.utils.data import ConcatDataset, Dataset

from .base import TaskDataset


class TorchTaskDataset(TaskDataset, Dataset):
    """The task dataset base class for all the torch-based task processors.

    This base class is enough for most cases, except there are procedures which can not be executed in
    preprocessors and Datasets like dataset merging.
    """

    def __init__(self,
                 datasets: Union[Any, List[Any]],
                 mode,
                 preprocessor=None,
                 **kwargs):
        TaskDataset.__init__(self, datasets, mode, preprocessor, **kwargs)
        self.trainer = None

    def __getitem__(self, index) -> Any:
        return self.prepare_sample(self._inner_dataset[index])

    def __len__(self):
        return len(self._inner_dataset)

    def prepare_dataset(self, datasets: Union[Any, List[Any]]) -> Any:
        """Prepare a dataset.

        User can process the input datasets in a whole dataset perspective.
        This method gives a default implementation of datasets merging, user can override this
        method to write custom logics.

        Args:
            datasets: The original dataset(s)

        Returns: A single dataset, which may be created after merging.

        """
        if isinstance(datasets, List):
            if len(datasets) == 1:
                return datasets[0]
            elif len(datasets) > 1:
                return ConcatDataset(datasets)
        else:
            return datasets

    def prepare_sample(self, data):
        """Preprocess the data fetched from the inner_dataset.

        If the preprocessor is None, the original data will be returned, else the preprocessor will be called.
        User can override this method to implement custom logics.

        Args:
            data: The data fetched from the dataset.

        Returns: The processed data.

        """
        return self.preprocessor(
            data) if self.preprocessor is not None else data

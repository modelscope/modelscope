# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union


class TaskDataset(ABC):
    """The task dataset base class for all the task specific dataset processors.
    """

    def __init__(self,
                 datasets: Union[Any, List[Any]],
                 mode,
                 preprocessor=None,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.preprocessor = preprocessor
        self._inner_dataset = self.prepare_dataset(datasets)

    @abstractmethod
    def prepare_dataset(self, datasets: Union[Any, List[Any]]) -> Any:
        """Prepare a dataset.

        User can process the input datasets in a whole dataset perspective.
        This method also helps to merge several datasets to one.

        Args:
            datasets: The original dataset(s)

        Returns: A single dataset, which may be created after merging.

        """
        pass

    @abstractmethod
    def prepare_sample(self, data):
        """Preprocess the data fetched from the inner_dataset.

        If the preprocessor is None, the original data will be returned, else the preprocessor will be called.
        User can override this method to implement custom logics.

        Args:
            data: The data fetched from the dataset.

        Returns: The processed data.

        """
        pass

# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, List, Union

import torch.utils.data
from torch.utils.data import ConcatDataset as TorchConcatDataset

from modelscope.utils.constant import ModeKeys


class TorchCustomDataset(torch.utils.data.Dataset):
    """The custom dataset base class for all the torch-based task processors.
    """

    def __init__(self,
                 datasets: Union[Any, List[Any]],
                 mode=ModeKeys.TRAIN,
                 preprocessor=None,
                 **kwargs):
        self.trainer = None
        self.mode = mode
        self.preprocessor = preprocessor
        self._inner_dataset = self.prepare_dataset(datasets)

    def __getitem__(self, index) -> Any:
        return self.preprocessor(
            self._inner_dataset[index]
        ) if self.preprocessor else self._inner_dataset[index]

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
                return TorchConcatDataset(datasets)
        else:
            return datasets

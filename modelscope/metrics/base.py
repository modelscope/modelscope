# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Dict


class Metric(ABC):
    """The metric base class for computing metrics.

    The subclasses can either compute a single metric like 'accuracy', or compute the
    complex metrics for a specific task with or without other Metric subclasses.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def add(self, outputs: Dict, inputs: Dict):
        """ Append logits and labels within an eval loop.

        Will be called after every batch finished to gather the model predictions and the labels.

        Args:
            outputs: The model prediction outputs.
            inputs: The mini batch inputs from the dataloader.

        Returns: None

        """
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the metrics after the eval finished.

        Will be called after the whole validation finished.

        Returns: The actual metric dict with standard names.

        """
        pass

    @abstractmethod
    def merge(self, other: 'Metric'):
        """ When using data parallel, the data required for different metric calculations

        are stored in their respective Metric classes,

        and we need to merge these data to uniformly calculate metric.

        Args:
            other: Another Metric instance.

        Returns: None

        """
        pass

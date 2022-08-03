# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Dict


class Metric(ABC):
    """The metric base class for computing metrics.

    The subclasses can either compute a single metric like 'accuracy', or compute the
    complex metrics for a specific task with or without other Metric subclasses.
    """

    def __init__(self, trainer=None, *args, **kwargs):
        self.trainer = trainer

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

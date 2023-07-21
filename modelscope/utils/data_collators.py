# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

from .logger import get_logger

logger = get_logger()


class RemoveColumnsCollator:
    """Remove specified columns from the input mini-batch, and convert them to attributes.

    For example: if columns_to_remove = ['id'], then user should call batch.id instead of batch['id'].

    Args:
        data_collator: An inner data collator to collate the mini-batch
        columns_to_remove(`List[str]`): The redundant columns to be removed from the mini-batch
        model_name(`Optional[str]`): An optional model name to print into log
        description(`Optional[str]`): An optional description to print into log
    """

    def __init__(
        self,
        data_collator,
        columns_to_remove: List[str],
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.data_collator = data_collator
        self.columns_to_remove = columns_to_remove
        self.description = description
        self.model_name = model_name
        self.message_logged = False

    def _remove_columns(self, feature: Mapping) -> Tuple[Mapping, Any]:
        if not isinstance(feature, Mapping):
            return feature, None
        if not self.message_logged and self.model_name:
            ignored_columns = list(
                set(feature.keys()) - set(self.columns_to_remove))
            if len(ignored_columns) > 0:
                dset_description = '' if self.description is None else f'in the {self.description} set'
                logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f"Legal columns: {', '.join(self.columns_to_remove)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`, "
                    ' you can safely ignore this message.')
                self.message_logged = True
        feature_clean = {
            k: v
            for k, v in feature.items() if k in self.columns_to_remove
        }
        feature_unused = {
            k: v
            for k, v in feature.items() if k not in self.columns_to_remove
        }
        return feature_clean, feature_unused

    def __call__(self, features: List[Mapping]):
        features_clean = []
        features_unused = []
        for feature in features:
            feature, feature_unused = self._remove_columns(feature)
            features_clean.append(feature)
            features_unused.append(feature_unused)
        data = OrderedDict(self.data_collator(features_clean))
        if features_unused[0] is not None:
            for key in features_unused[0].keys():
                setattr(data, key, [
                    feature_unused[key] for feature_unused in features_unused
                ])
        return data

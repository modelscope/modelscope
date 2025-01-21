# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.models.nlp import BertForTextRanking
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp_trainer import NlpEpochBasedTrainer
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class SentenceEmbeddingCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_length = 128
    tokenizer = None

    def __call__(self, features):
        qq = [f['query'] for f in features]
        dd = [f['docs'] for f in features]
        keys = qq[0].keys()
        qq = {k: [ele[k] for ele in qq] for k in keys}
        q_collated = self.tokenizer._tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt')
        keys = dd[0].keys()
        dd = {k: sum([ele[k] for ele in dd], []) for k in keys}
        d_collated = self.tokenizer._tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt')
        return {'query': q_collated, 'docs': d_collated}


@TRAINERS.register_module(module_name=Trainers.nlp_sentence_embedding_trainer)
class SentenceEmbeddingTrainer(NlpEpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Callable] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Preprocessor] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            **kwargs):

        super().__init__(
            model=model,
            cfg_file=cfg_file,
            cfg_modify_fn=cfg_modify_fn,
            arg_parse_fn=arg_parse_fn,
            data_collator=data_collator,
            preprocessor=preprocessor,
            optimizers=optimizers,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_revision=model_revision,
            **kwargs)

    def get_data_collator(self, data_collator, **kwargs):
        """Get the data collator for both training and evaluating.

        Args:
            data_collator: The input data_collator param.

        Returns:
            The train_data_collator and eval_data_collator, can be None.
        """
        if data_collator is None:
            data_collator = SentenceEmbeddingCollator(
                tokenizer=self.train_preprocessor.nlp_tokenizer,
                max_length=self.train_preprocessor.max_length)
        return super().get_data_collator(data_collator, **kwargs)

    def evauate(self):
        return {}

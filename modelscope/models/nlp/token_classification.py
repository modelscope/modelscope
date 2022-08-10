from abc import abstractmethod
from typing import Dict

import numpy as np
import torch
from torch import nn

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)
from .structbert import SbertPreTrainedModel

__all__ = ['SbertForTokenClassification']


class TokenClassification(TorchModel):
    """A token classification base class for all the fitted token classification models.
    """

    base_model_prefix: str = 'bert'

    def __init__(self, config, model_dir):
        super().__init__(model_dir)
        self.num_labels = config.num_labels
        self.config = config
        setattr(self, self.base_model_prefix, self.build_base_model())
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @abstractmethod
    def build_base_model(self):
        """Build the backbone model.

        Returns: the backbone instance.
        """
        pass

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix)

    def compute_loss(self, logits, labels, **kwargs):
        """Compute loss.

        For example, if backbone is pretrained model, there will be a 'attention_mask' parameter to skip
        useless tokens.

        Args:
            logits: The logits from the classifier
            labels: The labels
            **kwargs: Other input params.

        Returns: The loss.

        """
        pass

    def forward(self, **kwargs):
        labels = None
        if OutputKeys.LABEL in kwargs:
            labels = kwargs.pop(OutputKeys.LABEL)
        elif OutputKeys.LABELS in kwargs:
            labels = kwargs.pop(OutputKeys.LABELS)

        outputs = self.base_model(**kwargs)
        # base model should return the sequence_output as its first output
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss = self.compute_loss(logits, labels, **kwargs)
            return {OutputKeys.LOGITS: logits, OutputKeys.LOSS: loss}
        return {OutputKeys.LOGITS: logits}

    def postprocess(self, input: Dict[str, np.ndarray],
                    **kwargs) -> Dict[str, np.ndarray]:
        logits = input[OutputKeys.LOGITS]
        pred = torch.argmax(logits[0], dim=-1)
        pred = torch_nested_numpify(torch_nested_detach(pred))
        logits = torch_nested_numpify(torch_nested_detach(logits))
        rst = {OutputKeys.PREDICTIONS: pred, OutputKeys.LOGITS: logits}
        return rst


@MODELS.register_module(Tasks.word_segmentation, module_name=Models.structbert)
@MODELS.register_module(
    Tasks.token_classification, module_name=Models.structbert)
class SbertForTokenClassification(TokenClassification, SbertPreTrainedModel):
    """Sbert token classification model.

    Inherited from TokenClassification.
    """

    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r'pooler']

    def __init__(self, config, model_dir):
        if hasattr(config, 'base_model_prefix'):
            SbertForTokenClassification.base_model_prefix = config.base_model_prefix
        super().__init__(config, model_dir)

    def build_base_model(self):
        from .structbert import SbertModel
        return SbertModel(self.config, add_pooling_layer=False)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                **kwargs):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels)

    def compute_loss(self, logits, labels, attention_mask=None, **kwargs):
        """Compute the loss with an attention mask.

        @param logits: The logits output from the classifier.
        @param labels: The labels.
        @param attention_mask: The attention_mask.
        @param kwargs: Unused input args.
        @return: The loss
        """
        loss_fct = nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels))
            return loss_fct(active_logits, active_labels)
        else:
            return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        @param kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels not supplied.
                                    If num_labels is not found, the model will use the default setting (2 classes).
        @return: The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """
        model_dir = kwargs.get('model_dir')
        num_labels = kwargs.get('num_labels')
        if num_labels is None:
            label2id = parse_label_mapping(model_dir)
            if label2id is not None and len(label2id) > 0:
                num_labels = len(label2id)

        model_args = {} if num_labels is None else {'num_labels': num_labels}
        return super(SbertPreTrainedModel,
                     SbertForTokenClassification).from_pretrained(
                         pretrained_model_name_or_path=kwargs.get('model_dir'),
                         model_dir=kwargs.get('model_dir'),
                         **model_args)

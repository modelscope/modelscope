from abc import abstractmethod

from torch import nn

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.structbert import SbertPreTrainedModel
from modelscope.models.nlp.veco import \
    VecoForSequenceClassification as VecoForSequenceClassificationTransform
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

__all__ = ['SbertForSequenceClassification', 'VecoForSequenceClassification']


class SequenceClassificationBase(TorchModel):
    """A sequence classification base class for all the fitted sequence classification models.
    """
    base_model_prefix: str = 'bert'

    def __init__(self, config, model_dir):
        super().__init__(model_dir)
        self.num_labels = config.num_labels
        self.config = config
        setattr(self, self.base_model_prefix, self.build_base_model())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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

    def forward(self, **kwargs):
        labels = None
        if OutputKeys.LABEL in kwargs:
            labels = kwargs.pop(OutputKeys.LABEL)
        elif OutputKeys.LABELS in kwargs:
            labels = kwargs.pop(OutputKeys.LABELS)

        outputs = self.base_model.forward(**kwargs)

        # backbone model should return pooled_output as its second output
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {OutputKeys.LOGITS: logits, OutputKeys.LOSS: loss}
        return {OutputKeys.LOGITS: logits}

    def postprocess(self, input, **kwargs):
        logits = input[OutputKeys.LOGITS]
        probs = torch_nested_numpify(torch_nested_detach(logits.softmax(-1)))
        pred = torch_nested_numpify(torch_nested_detach(logits.argmax(-1)))
        logits = torch_nested_numpify(torch_nested_detach(logits))
        res = {
            OutputKeys.PREDICTIONS: pred,
            OutputKeys.PROBABILITIES: probs,
            OutputKeys.LOGITS: logits
        }
        return res


@MODELS.register_module(
    Tasks.sentence_similarity, module_name=Models.structbert)
@MODELS.register_module(
    Tasks.sentiment_classification, module_name=Models.structbert)
@MODELS.register_module(Tasks.nli, module_name=Models.structbert)
@MODELS.register_module(
    Tasks.zero_shot_classification, module_name=Models.structbert)
class SbertForSequenceClassification(SequenceClassificationBase,
                                     SbertPreTrainedModel):
    """Sbert sequence classification model.

    Inherited from SequenceClassificationBase.
    """
    base_model_prefix: str = 'bert'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, model_dir):
        if hasattr(config, 'base_model_prefix'):
            SbertForSequenceClassification.base_model_prefix = config.base_model_prefix
        super().__init__(config, model_dir)

    def build_base_model(self):
        from .structbert import SbertModel
        return SbertModel(self.config, add_pooling_layer=True)

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
                     SbertForSequenceClassification).from_pretrained(
                         pretrained_model_name_or_path=kwargs.get('model_dir'),
                         model_dir=kwargs.get('model_dir'),
                         **model_args)


@MODELS.register_module(Tasks.sentence_similarity, module_name=Models.veco)
@MODELS.register_module(
    Tasks.sentiment_classification, module_name=Models.veco)
@MODELS.register_module(Tasks.nli, module_name=Models.veco)
class VecoForSequenceClassification(TorchModel,
                                    VecoForSequenceClassificationTransform):
    """Veco sequence classification model.

    Inherited from VecoForSequenceClassification and TorchModel, so this class can be registered into the model set.
    This model cannot be inherited from SequenceClassificationBase, because Veco/XlmRoberta's classification structure
    is different.
    """

    def __init__(self, config, model_dir):
        super().__init__(model_dir)
        VecoForSequenceClassificationTransform.__init__(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs):
        return VecoForSequenceClassificationTransform.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels)

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        @param kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels not supplied.
                                    If num_labels is not found, the model will use the default setting (2 classes).
        @return: The loaded model, which is initialized by veco.VecoForSequenceClassification.from_pretrained
        """

        model_dir = kwargs.get('model_dir')
        num_labels = kwargs.get('num_labels')
        if num_labels is None:
            label2id = parse_label_mapping(model_dir)
            if label2id is not None and len(label2id) > 0:
                num_labels = len(label2id)

        model_args = {} if num_labels is None else {'num_labels': num_labels}
        return super(VecoForSequenceClassificationTransform,
                     VecoForSequenceClassification).from_pretrained(
                         pretrained_model_name_or_path=kwargs.get('model_dir'),
                         model_dir=kwargs.get('model_dir'),
                         **model_args)

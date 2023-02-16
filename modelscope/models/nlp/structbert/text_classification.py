# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionTextClassificationModelOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .adv_utils import compute_adv_loss
from .backbone import SbertModel, SbertPreTrainedModel
from .configuration import SbertConfig

logger = logging.get_logger()


@MODELS.register_module(
    Tasks.text_classification, module_name=Models.structbert)
@MODELS.register_module(Tasks.nli, module_name=Models.structbert)
@MODELS.register_module(
    Tasks.sentiment_classification, module_name=Models.structbert)
@MODELS.register_module(
    Tasks.sentence_similarity, module_name=Models.structbert)
@MODELS.register_module(
    Tasks.zero_shot_classification, module_name=Models.structbert)
class SbertForSequenceClassification(SbertPreTrainedModel):
    r"""StructBERT Model transformer with a sequence classification/regression head on top
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Preprocessor:
        This is the text classification model of StructBERT, the preprocessor of this model
        is `modelscope.preprocessors.TextClassificationTransformersPreprocessor`.

    Trainer:
        This model is a normal PyTorch model, and can be trained by variable trainers, like EpochBasedTrainer,
        NlpEpochBasedTrainer, or trainers from other frameworks.
        The preferred trainer in ModelScope is NlpEpochBasedTrainer.

    Parameters:
        config (:class:`~modelscope.models.nlp.structbert.SbertConfig`): Model configuration class with
            all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
    """

    def __init__(self, config: SbertConfig, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if self.config.adv_grad_factor is None:
            logger.warning(
                'Adv parameters not set, skipping compute_adv_loss.')

        SbertForSequenceClassification.base_model_prefix = getattr(
            config, 'base_model_prefix',
            SbertForSequenceClassification.base_model_prefix)
        setattr(self, self.base_model_prefix, SbertModel(config))
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def _forward_call(self, **kwargs):
        outputs = self.base_model(**kwargs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs['logits'] = logits
        outputs.kwargs = kwargs
        return outputs

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
                return_dict=None,
                *args,
                **kwargs):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using :class:`~modelscope.models.nlp.structbert.SbertTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
                1]``:

                - 0 corresponds to a `sentence A` token,
                - 1 corresponds to a `sentence B` token.

            position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                ``[0, config.max_position_embeddings - 1]``.

            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`,
                `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
                `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.ModelOutput` instead of a plain tuple.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed
                (Mean-Square loss), If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Returns `modelscope.outputs.AttentionTextClassificationModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
            >>> # Call the model, return some tensors
            >>> print(model(**preprocessor(('这是个测试', '这也是个测试'))))
            >>> # Call the pipeline
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('text-classification', model=model, preprocessor=preprocessor)
            >>> print(pipeline_ins(('这是个测试', '这也是个测试')))
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            logger.error('Return tuple in sbert is not supported now.')
        outputs = self._forward_call(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        return self.compute_loss(outputs, labels, **outputs.kwargs)

    def compute_loss(self, outputs, labels, **kwargs):
        logits = outputs.logits
        embedding_output = outputs.embedding_output
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long
                                              or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
                if self.config.adv_grad_factor is not None and self.training:
                    loss = compute_adv_loss(
                        embedding=embedding_output,
                        model=self._forward_call,
                        ori_logits=logits,
                        ori_loss=loss,
                        adv_bound=self.config.adv_bound,
                        adv_grad_factor=self.config.adv_grad_factor,
                        sigma=self.config.sigma,
                        **kwargs)
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return AttentionTextClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

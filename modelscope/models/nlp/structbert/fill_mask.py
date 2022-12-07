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
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionFillMaskModelOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .backbone import SbertModel, SbertPreTrainedModel
from .configuration import SbertConfig

logger = logging.get_logger()


class SbertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SbertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = SbertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SbertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = SbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SbertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = SbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


@MODELS.register_module(Tasks.fill_mask, module_name=Models.structbert)
class SbertForMaskedLM(SbertPreTrainedModel):
    r"""StructBERT Model with a `language modeling` head on top.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Preprocessor:
        This is the fill_mask model of StructBERT, the preprocessor of this model
        is `modelscope.preprocessors.FillMaskTransformersPreprocessor`.

    Parameters:
        config (:class:`~modelscope.models.nlp.structbert.SbertConfig`): Model configuration class with
            all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
    """

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [
        r'position_ids', r'predictions.decoder.bias'
    ]

    def __init__(self, config: SbertConfig, **kwargs):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                'If you want to use `SbertForMaskedLM` make sure `config.is_decoder=False` for '
                'bi-directional self-attention.')

        self.bert = SbertModel(config)
        self.cls = SbertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~modelscope.models.nlp.structbert.SbertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__

        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.ModelOutput` instead of a plain tuple.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:
            Returns `modelscope.outputs.AttentionFillMaskModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor, FillMaskTransformersPreprocessor
            >>> model = Model.from_pretrained('damo/nlp_structbert_fill-mask_chinese-large')
            >>> preprocessor = FillMaskTransformersPreprocessor('damo/nlp_structbert_fill-mask_chinese-large')
            >>> # Call the model, return some tensors
            >>> print(model(**preprocessor('你师父差得动你，你师父可[MASK]不动我。')))
            >>> # Call the pipeline
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('fill-mask', model=model, preprocessor=preprocessor)
            >>> print(pipeline_ins('你师父差得动你，你师父可[MASK]不动我。'))
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:-1]
            return ((masked_lm_loss, )
                    + output) if masked_lm_loss is not None else output

        return AttentionFillMaskModelOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            input_ids=input_ids,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, 'The PAD token should be defined for generation'
        attention_mask_zero = attention_mask.new_zeros(
            (attention_mask.shape[0], 1))
        attention_mask = torch.cat([attention_mask, attention_mask_zero],
                                   dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

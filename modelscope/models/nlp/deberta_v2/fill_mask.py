# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
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

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionFillMaskModelOutput
from modelscope.utils.constant import Tasks
from .backbone import DebertaV2Model, DebertaV2PreTrainedModel


# Copied from transformers.models.deberta.modeling_deberta.DebertaForMaskedLM with Deberta->DebertaV2
@MODELS.register_module(Tasks.fill_mask, module_name=Models.deberta_v2)
class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    r"""DeBERTa_v2 Model with a `language modeling` head on top.

    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Preprocessor:
        This is the fill_mask model of Deberta_v2, the preprocessor of this model
        is `modelscope.preprocessors.FillMaskTransformersPreprocessor`.

    Parameters:
        config (`DebertaV2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration.
    """

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [
        r'position_ids', r'predictions.decoder.bias'
    ]

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AttentionFillMaskModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `('batch_size, sequence_length')`):
                Indices of input sequence tokens in the vocabulary.

            attention_mask (`torch.FloatTensor` of shape `('batch_size, sequence_length')`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            token_type_ids (`torch.LongTensor` of shape `('batch_size, sequence_length')`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
                1]`:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

            position_ids (`torch.LongTensor` of shape `('batch_size, sequence_length')`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range `[0, config.max_position_embeddings - 1]`.

            inputs_embeds (`torch.FloatTensor` of shape `('batch_size, sequence_length', hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert *input_ids* indices into associated
                vectors than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a dataclass instead of a plain tuple.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            Returns `modelscope.outputs.AttentionFillMaskModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_debertav2_fill-mask_chinese-lite')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_debertav2_fill-mask_chinese-lite')
            >>> # Call the model, return some tensors
            >>> print(model(**preprocessor('你师父差得动你，你师父可[MASK]不动我。')))
            >>> # Call the pipeline
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('fill-mask', model=model, preprocessor=preprocessor)
            >>> print(pipeline_ins('你师父差得动你，你师父可[MASK]不动我。'))
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
            output = (prediction_scores, ) + outputs[1:]
            return ((masked_lm_loss, )
                    + output) if masked_lm_loss is not None else output

        return AttentionFillMaskModelOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            input_ids=input_ids,
            attentions=outputs.attentions,
            hidden_states=outputs.hidden_states)


# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaV2PredictionHeadTransform(nn.Module):

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


# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaV2LMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = DebertaV2PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaV2OnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

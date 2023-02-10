# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
from transformers.modeling_utils import PreTrainedModel

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import (OutputKeys, TextGenerationModelOutput,
                                TokenGeneratorOutput)
from modelscope.utils.constant import Tasks

__all__ = ['ModelForTextGeneration']


@MODELS.register_module(
    Tasks.text_generation, module_name=TaskModels.text_generation)
class ModelForTextGeneration(SingleBackboneTaskModelBase, PreTrainedModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        if 'base_model_prefix' in kwargs:
            self._base_model_prefix = kwargs['base_model_prefix']

        self.build_backbone(self.backbone_cfg)
        self.build_head(self.head_cfg)
        if self.config.get('shared_embedding', False):
            input_embeddings = self.backbone.get_input_embeddings()
            output_embeddings = self.head.get_output_embeddings()
            output_embeddings.weight = input_embeddings.weight

    def forward(self, **input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # backbone do not need labels, only head need for loss compute
        labels = input.pop(OutputKeys.LABELS, None)

        backbone_outputs = super().forward(input)
        hidden_states = backbone_outputs[0]

        logits = self.head.forward(hidden_states)
        loss = None
        if labels is not None:
            input[OutputKeys.LABELS] = labels
            loss = self.compute_loss(logits, labels)
        return TextGenerationModelOutput(logits=logits, loss=loss)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            'input_ids': input_ids,
            'past_key_values': past,
            'use_cache': kwargs.get('use_cache'),
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }

    def generate(self, inputs, *args, **kwargs):
        input_ids = inputs['input_ids'] if isinstance(inputs, Dict) else inputs
        generate_output = super().generate(input_ids, *args, **kwargs)
        if isinstance(generate_output, Dict):
            return TokenGeneratorOutput(
                sequences=generate_output.sequences,
                scores=generate_output.scores,
                attentions=generate_output.attentions,
                hidden_states=generate_output.hidden_states)
        else:
            return TokenGeneratorOutput(sequences=generate_output)

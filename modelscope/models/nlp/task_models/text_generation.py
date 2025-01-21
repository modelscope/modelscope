# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
import torch
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

    def forward(self, **inputs) -> Dict[str, np.ndarray]:
        # backbone do not need labels, only head need for loss compute
        labels = inputs.pop(OutputKeys.LABELS, None)

        backbone_outputs = super().forward(inputs)
        hidden_states = backbone_outputs[0]

        logits = self.head.forward(hidden_states)
        loss = None
        if labels is not None:
            inputs[OutputKeys.LABELS] = labels
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

    def generate(self, inputs, temperature=1.0, **kwargs):
        tokens = inputs['input_ids'] if isinstance(inputs, Dict) else inputs
        top_k = kwargs.pop('top_k',
                           self.config.top_k if 'top_k' in self.config else 1)
        top_p = kwargs.pop('top_p',
                           self.config.top_p if 'top_p' in self.config else 0.)
        max_length = kwargs.pop('max_length', self.config.max_length)

        batch_size = tokens.size(0)
        lengths = kwargs.pop(
            'prompt_length',
            torch.tensor([tokens.size(1)], device=tokens.device))

        min_prompt_length = lengths.min().item()
        max_sequence_length = max_length

        # If the context is too big, this happens
        if min_prompt_length >= max_sequence_length:
            raise ValueError('context length too large')

        pad_length = max_sequence_length - tokens.size(1)
        if pad_length > 0:
            pads = torch.zeros(
                batch_size, pad_length, device=tokens.device).long()
            tokens = torch.cat((tokens, pads), dim=-1)

        # Added termination_id to support the case that we want to terminate the
        # generation once that id is generated.
        termination_id = self.config.eos_token_id

        # Whether we have reached a termination id.
        is_generation_done = torch.zeros(
            batch_size, dtype=torch.uint8, device=tokens.device)

        with torch.no_grad():
            for context_length in range(min_prompt_length,
                                        max_sequence_length):

                # Pick the slice that we need to pass through the network.
                tokens2use = tokens[:, :context_length]

                # logits will be meanigful only in the last pipeline stage.
                logits = self(input_ids=tokens2use).logits

                # Sample.
                last_token_logits = logits[:, -1, :]
                new_sample = sample(
                    last_token_logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    vocab_size=self.head_cfg.vocab_size)

                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]

                done_token = (new_sample == termination_id).byte() & \
                    started.byte()

                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)

                if done:
                    break

        tokens = tokens[:, :(context_length + 1)]
        return TokenGeneratorOutput(sequences=tokens)


def sample(logits, top_k=0, top_p=0.0, temperature=1.0, vocab_size=None):
    """ Sample and generate a token.
    Note: logits has the dimension [b, v] where b is the batch size
          and v is the vocabulary size.
    If vocab_size is provided, we will make sure the sample that is
    generated is in [0, vocab-size). This will avoid out of vocabulary
    generations due to padding.
    """

    # Check logits for consistency.
    assert logits.ndim == 2, 'expected the logits to be of [b, v] shape.'

    # Greedy is just simple argmax.
    if top_k == 1:
        assert top_p == 0.0, 'cannot set both greedy and top-p samplings.'
        samples = torch.argmax(logits, dim=-1)

    # Top-k or top-p sampling.
    else:
        # Clone so we do not modify the inputs,
        logits = logits.clone()
        # Apply temperature in place.
        if temperature != 1.0:
            logits.div_(temperature)

        if top_k > 1:
            top_p == 0.0
            assert top_k <= logits.size(1), 'top-k is larger than logit size.'
            if vocab_size:
                assert top_k < vocab_size, 'top-k is larger than vocab size.'
            modify_logits_for_top_k_filtering(logits, top_k)

        elif top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
            modify_logits_for_top_p_filtering(logits, top_p)

        # After filtering, we need to recalculate the distribution.
        probs = logits.softmax(dim=-1)
        samples = torch.multinomial(probs, num_samples=1).view(-1)

    # If vocab size is provided, make sure the samples are in
    # in the range [0, vocab-size).
    if vocab_size:
        samples = torch.clamp(samples, min=0, max=(vocab_size - 1))

    return samples


def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf."""

    filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(filter_, float('-Inf'))


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Filteration based on the cumulative sum.
    filter_ = cumulative_probs > top_p
    # This shift by 1 is weird and I cannot justify it. This existed
    # in the original implementation:
    #   https://github.com/ari-holtzman/degen/blob/master/gen.py
    # and I guess it is needed so keeping it for now.
    filter_[:, 1:] = filter_[:, :-1].clone()
    # Make sure we at least have one token to select from.
    filter_[..., 0] = 0

    # Fill in the filtered part
    filter_ = filter_.scatter(1, sorted_indices, filter_)
    logits.masked_fill_(filter_, float('-Inf'))

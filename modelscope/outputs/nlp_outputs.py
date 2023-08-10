from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from modelscope.outputs.outputs import ModelOutputBase

Tensor = Union['torch.Tensor', 'tf.Tensor']


@dataclass
class BackboneModelOutput(ModelOutputBase):
    """The output class for text classification models.

    Args:
        last_hidden_state (`Tensor`, *optional*): Sequence of hidden-states at
            the output of the last layer of the model.
        pooler_output (`Tensor`, *optional*) The tensor of the pooled hidden state.
        hidden_states (`Tensor`, *optional*) Hidden-states of the model at
            the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Tensor = None
    pooler_output: Tensor = None
    hidden_states: Tensor = None


@dataclass
class AttentionBackboneModelOutput(BackboneModelOutput):
    """The output class for backbones of attention based models.

    Args:
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when
        `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when
        `output_attentions=True` and `config.add_cross_attention=True` is passed
        or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned
        when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`,
            with each tuple having 2 tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length,
            embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the
            self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that
            can be used (see `past_key_values` input) to speed up sequential
            decoding.
    """
    attentions: Tensor = None
    past_key_values: Tensor = None
    cross_attentions: Tensor = None


@dataclass
class Seq2SeqModelOutput(ModelOutputBase):
    """
    Base class for model encoder's outputs that also contains : pre-computed
    hidden states that can speed up sequential decoding.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size,
        sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the
            decoder of the model.

            If `past_key_values` is used only the last hidden-state of the
            sequences of shape `(batch_size, 1, hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned
        when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`,
            with each tuple having 2 tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape `(batch_size, num_heads, encoder_sequence_length,
            embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the
            self-attention blocks and in the cross-attention blocks) that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the
            optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used
            to compute the weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when
        `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the
            cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size,
        sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the
            encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the
            optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used
            to compute the weighted average in the self-attention heads.
    """

    last_hidden_state: Tensor = None
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tensor]] = None
    decoder_attentions: Optional[Tuple[Tensor]] = None
    cross_attentions: Optional[Tuple[Tensor]] = None
    encoder_last_hidden_state: Optional[Tensor] = None
    encoder_hidden_states: Optional[Tuple[Tensor]] = None
    encoder_attentions: Optional[Tuple[Tensor]] = None


@dataclass
class FaqQuestionAnsweringOutput(ModelOutputBase):
    """The output class for faq QA models.
    """

    scores: Tensor = None
    labels: Tensor = None
    loss: Tensor = None
    logits: Tensor = None


@dataclass
class FeatureExtractionOutput(ModelOutputBase):
    """The output class for feature extraction models.
    """

    text_embedding: Tensor = None


@dataclass
class FillMaskModelOutput(ModelOutputBase):
    """The output class for fill mask models.

    Args:
        logits (`Tensor`): The logits output of the model.
        loss (`Tensor`, *optional*) The loss of the model, available when training.
        input_ids (`Tensor`, *optional*) The input id tensor fed into the model.
        hidden_states (`Tensor`, *optional*) Hidden-states of the model at the
            output of each layer plus the optional initial embedding outputs.
    """

    logits: Tensor = None
    loss: Tensor = None
    input_ids: Tensor = None
    hidden_states: Tensor = None


@dataclass
class AttentionFillMaskModelOutput(FillMaskModelOutput):
    """The output class for the fill mask and attention based models.

    Args:
        attentions (`tuple(Tensor)`, *optional* Attentions weights after the
        attention softmax, used to compute the weighted average in the
        self-attention heads.
    """
    attentions: Tensor = None


@dataclass
class InformationExtractionOutput(ModelOutputBase):
    """The output class for information extraction models.
    """

    spo_list: np.ndarray = None


@dataclass
class Seq2SeqLMOutput(ModelOutputBase):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when
        `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,
        config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each
            vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned
        when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`,
            with each tuple having 2 tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape `(batch_size, num_heads, encoder_sequence_length,
            embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the
            self-attention blocks and in the cross-attention blocks) that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the
            initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used
            to compute the weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when
        `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the
            cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size,
        sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the
            encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the
            initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned
        when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used
            to compute the weighted average in the self-attention heads.
    """

    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tensor]] = None
    decoder_attentions: Optional[Tuple[Tensor]] = None
    cross_attentions: Optional[Tuple[Tensor]] = None
    encoder_last_hidden_state: Optional[Tensor] = None
    encoder_hidden_states: Optional[Tuple[Tensor]] = None
    encoder_attentions: Optional[Tuple[Tensor]] = None


@dataclass
class TextClassificationModelOutput(ModelOutputBase):
    """The output class for text classification models.

    Args:
        logits (`Tensor`): The logits output of the model. loss (`Tensor`,
        *optional*) The loss of the model, available when training.
        hidden_states (`Tensor`, *optional*) Hidden-states of the model at the
        output of each layer plus the optional initial embedding outputs.
    """

    logits: Tensor = None
    loss: Tensor = None


@dataclass
class AttentionTextClassificationModelOutput(TextClassificationModelOutput):
    """The output class for backbones of attention based models.

    Args:
        attentions (`tuple(Tensor)`, *optional* Attentions weights after the
        attention softmax, used to compute the weighted average in the
        self-attention heads.
    """
    attentions: Tensor = None
    hidden_states: Tensor = None
    past_key_values: Tensor = None


@dataclass
class TextErrorCorrectionOutput(ModelOutputBase):
    """The output class for information extraction models.
    """

    predictions: np.ndarray = None


@dataclass
class WordAlignmentOutput(ModelOutputBase):
    """The output class for word alignment models.
    """

    predictions: List = None


@dataclass
class TextGenerationModelOutput(ModelOutputBase):
    """The output class for text generation models.

    Args:
        logits (`Tensor`): The logits output of the model. loss (`Tensor`,
        *optional*) The loss of the model, available when training.
        hidden_states (`Tensor`, *optional*) Hidden-states of the model at the
        output of each layer plus the optional initial embedding outputs.
    """

    logits: Tensor = None
    loss: Tensor = None


@dataclass
class AttentionTextGenerationModelOutput(TextGenerationModelOutput):
    """The output class for text generation of attention based models.

    Args:
        logits (`Tensor`): The logits output of the model. loss (`Tensor`,
        *optional*) The loss of the model, available when training.
        hidden_states (`Tensor`, *optional*) Hidden-states of the model at the
        output of each layer plus the optional initial embedding outputs.
    """
    attentions: Tensor = None
    hidden_states: Tensor = None
    past_key_values: Tensor = None


@dataclass
class TokenGeneratorOutput(ModelOutputBase):
    """
    The output class for generate method of text generation models.


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`
        is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`
        is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(num_return_sequences*batch_size, num_heads, generated_length,
            sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`
        is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: Tensor = None
    scores: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tuple[Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[Tensor]]] = None


@dataclass
class TokenClassificationModelOutput(ModelOutputBase):
    """The output class for token classification models.
        logits (`Tensor`): The logits output of the model.
        loss (`Tensor`, *optional*) The loss of the model, available when training.
        predictions: A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        offset_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
        sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the sentence.
            Selected in the range ``[0, sequence_length - 1]``.
    """

    logits: Tensor = None
    loss: Tensor = None
    offset_mapping: Tensor = None
    predictions: Tensor = None
    label_mask: Tensor = None


@dataclass
class AttentionTokenClassificationModelOutput(TokenClassificationModelOutput):
    """The output class for backbones of attention based models.

    Args:
        attentions (`tuple(Tensor)`, *optional* Attentions weights after the attention softmax,
        used to compute the weighted average in the self-attention heads.
    """
    attentions: Tensor = None
    hidden_states: Tensor = None


@dataclass
class DialogueUserSatisfactionEstimationModelOutput(ModelOutputBase):
    """The output class for user satisfaction estimation.

    Args:
        logits (`Tensor`): The logits output of the model.
    """
    logits: Tensor = None


@dataclass
class SentencEmbeddingModelOutput(ModelOutputBase):
    """The output class for text classification models.

    Args:
        query_embs (`Tensor`, *optional*): The tensor of the query embeddings.
        doc_embs (`Tensor`, *optional*) Then tensor of the doc embeddings.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*): Sentence Embedding modeling loss.
    """

    query_embeddings: Tensor = None
    doc_embeddings: Tensor = None
    loss: Tensor = None


@dataclass
class TranslationEvaluationOutput(ModelOutputBase):
    """The output class for translation evaluation models.
    """

    score: Tensor = None
    loss: Tensor = None
    input_format: List[str] = None


@dataclass
class MachineReadingComprehensionOutput(ModelOutputBase):
    """The output class for machine reading comprehension models.

    Args:
        loss (`Tensor`, *optional*): The training loss of the current batch
        match_loss (`Tensor`, *optinal*): The match loss of the current batch
        span_logits (`Tensor`): The logits of the span matrix output by the model
        hidden_states (`Tuple[Tensor]`, *optinal*): The hidden states output by the model
        attentions (`Tuple[Tensor]`, *optinal*):  The attention scores output by the model
        input_ids (`Tensor`): The token ids of the input sentence

    """

    loss: Optional[Tensor] = None
    match_loss: Optional[Tensor] = None
    span_logits: Tensor = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None
    input_ids: Tensor = None

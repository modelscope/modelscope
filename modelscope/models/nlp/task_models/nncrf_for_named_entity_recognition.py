# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors. All rights reserved.
# The CRF implementation borrows mostly from AllenNLP CRF module (https://github.com/allenai/allennlp)
# and pytorch-crf (https://github.com/kmkurn/pytorch-crf) with some modifications.

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionTokenClassificationModelOutput
from modelscope.utils.constant import ModelFile, Tasks

__all__ = [
    'TransformerCRFForNamedEntityRecognition',
    'LSTMCRFForNamedEntityRecognition', 'LSTMCRFForWordSegmentation',
    'LSTMCRFForPartOfSpeech'
]


class SequenceLabelingForNamedEntityRecognition(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model = self.init_model(model_dir, *args, **kwargs)

        model_ckpt = os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE)
        self.model.load_state_dict(
            torch.load(model_ckpt, map_location=torch.device('cpu')))

    def init_model(self, model_dir, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(
        self,
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
        offset_mapping=None,
        label_mask=None,
    ) -> Dict[str, Any]:
        r"""
        Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~modelscope.models.nlp.structbert.SbertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

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
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        offset_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
        sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the sentence.
            Selected in the range ``[0, sequence_length - 1]``.
        label_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
        sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask
            values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        Returns:
            Returns `modelscope.outputs.AttentionTokenClassificationModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_structbert_word-segmentation_chinese-base')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_word-segmentation_chinese-base')
            >>> print(model(**preprocessor(('This is a test', 'This is also a test'))))
        """
        input_tensor = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_mask': label_mask,
        }
        output = {
            'offset_mapping': offset_mapping,
            **input_tensor,
            **self.model(input_tensor)
        }
        return output

    def postprocess(self, input: Dict[str, Any], **kwargs):
        predicts = self.model.decode(input)
        offset_mapping = input.get('offset_mapping')
        mask = input.get('label_mask')

        # revert predicts to original position with respect of label mask
        masked_predict = torch.zeros_like(predicts)
        for i in range(len(mask)):
            masked_lengths = mask[i].sum(-1).long().cpu().item()
            selected_predicts = torch.narrow(
                predicts[i], 0, 0,
                masked_lengths)  # index_select only move loc, not resize
            mask_position = mask[i].byte()
            masked_predict[i][mask_position] = selected_predicts
        predicts = masked_predict

        return AttentionTokenClassificationModelOutput(
            loss=None,
            logits=None,
            hidden_states=None,
            attentions=None,
            label_mask=mask,
            offset_mapping=offset_mapping,
            predictions=predicts,
        )


@MODELS.register_module(
    Tasks.named_entity_recognition, module_name=Models.tcrf)
class TransformerCRFForNamedEntityRecognition(
        SequenceLabelingForNamedEntityRecognition):
    """This model wraps the TransformerCRF model to register into model sets.
    """

    def init_model(self, model_dir, *args, **kwargs):
        self.config = AutoConfig.from_pretrained(model_dir)
        num_labels = self.config.num_labels

        model = TransformerCRF(model_dir, num_labels)
        return model


@MODELS.register_module(Tasks.word_segmentation, module_name=Models.tcrf_wseg)
class TransformerCRFForWordSegmentation(TransformerCRFForNamedEntityRecognition
                                        ):
    """This model wraps the TransformerCRF model to register into model sets.
    """
    pass


@MODELS.register_module(
    Tasks.named_entity_recognition, module_name=Models.lcrf)
class LSTMCRFForNamedEntityRecognition(
        SequenceLabelingForNamedEntityRecognition):
    """This model wraps the LSTMCRF model to register into model sets.
    """

    def init_model(self, model_dir, *args, **kwargs):
        self.config = AutoConfig.from_pretrained(model_dir)
        vocab_size = self.config.vocab_size
        embed_width = self.config.embed_width
        num_labels = self.config.num_labels
        lstm_hidden_size = self.config.lstm_hidden_size

        model = LSTMCRF(vocab_size, embed_width, num_labels, lstm_hidden_size)
        return model


@MODELS.register_module(Tasks.word_segmentation, module_name=Models.lcrf_wseg)
@MODELS.register_module(Tasks.word_segmentation, module_name=Models.lcrf)
class LSTMCRFForWordSegmentation(LSTMCRFForNamedEntityRecognition):
    pass


@MODELS.register_module(Tasks.part_of_speech, module_name=Models.lcrf)
class LSTMCRFForPartOfSpeech(LSTMCRFForNamedEntityRecognition):
    pass


class TransformerCRF(nn.Module):
    """A transformer based model to NER tasks.

    This model will use transformers' backbones as its backbone.
    """

    def __init__(self, model_dir, num_labels, **kwargs):
        super(TransformerCRF, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_dir)
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, inputs):
        embed = self.encoder(
            inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        logits = self.linear(embed)

        if 'label_mask' in inputs:
            mask = inputs['label_mask']
            masked_lengths = mask.sum(-1).long()
            masked_logits = torch.zeros_like(logits)
            for i in range(len(mask)):
                masked_logits[
                    i, :masked_lengths[i], :] = logits[i].masked_select(
                        mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
            logits = masked_logits

        outputs = {'logits': logits}
        return outputs

    def decode(self, inputs):
        seq_lens = inputs['label_mask'].sum(-1).long()
        mask = torch.arange(
            inputs['label_mask'].shape[1],
            device=seq_lens.device)[None, :] < seq_lens[:, None]
        predicts = self.crf.decode(inputs['logits'], mask=mask).squeeze(0)
        return predicts


class LSTMCRF(nn.Module):
    """
    A standard bilstm-crf model for fast prediction.
    """

    def __init__(self,
                 vocab_size,
                 embed_width,
                 num_labels,
                 lstm_hidden_size=100,
                 **kwargs):
        super(LSTMCRF, self).__init__()
        self.embedding = Embedding(vocab_size, embed_width)
        self.lstm = nn.LSTM(
            embed_width,
            lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True)
        self.ffn = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, inputs):
        embedding = self.embedding(inputs['input_ids'])
        lstm_output, _ = self.lstm(embedding)
        logits = self.ffn(lstm_output)

        if 'label_mask' in inputs:
            mask = inputs['label_mask']
            masked_lengths = mask.sum(-1).long()
            masked_logits = torch.zeros_like(logits)
            for i in range(len(mask)):
                masked_logits[
                    i, :masked_lengths[i], :] = logits[i].masked_select(
                        mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
            logits = masked_logits

        outputs = {'logits': logits}
        return outputs

    def decode(self, inputs):
        seq_lens = inputs['label_mask'].sum(-1).long()
        mask = torch.arange(
            inputs['label_mask'].shape[1],
            device=seq_lens.device)[None, :] < seq_lens[:, None]
        predicts = self.crf.decode(inputs['logits'], mask=mask).squeeze(0)
        return predicts


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm

    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self,
                emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self,
               emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2],
                dtype=torch.uint8,
                device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self,
                  emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}'
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}'
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self,
                        emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long,
                                  device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long,
                              device=device)
        oor_tag = torch.full((seq_length, batch_size),
                             pad_tag,
                             dtype=torch.long,
                             device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
            end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long,
                                    device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(
            self,
            emissions: torch.FloatTensor,
            mask: torch.ByteTensor,
            nbest: int,
            pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags, nbest),
            dtype=torch.long,
            device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long,
                              device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest),
                             pad_tag,
                             dtype=torch.long,
                             device=device)

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(
                    1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1,
                                                  self.num_tags).topk(
                                                      nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1),
                                next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices,
                                  oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
            end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long,
                                    device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
                         .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1,
                                     best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr,
                           oor_tag).permute(2, 1, 0)


class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_width):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_width)

    def forward(self, input_ids):
        return self.embedding(input_ids)

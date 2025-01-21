# Copyright 2020 The HuggingFace Inc. team
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

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Iterable, List, Optional, Tuple

import torch

PROCESS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        next_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Current scores of the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_tokens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            :obj:`input_ids` of the tokens corresponding to the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_indices (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Beam indices indicating to which beam hypothesis the :obj:`next_tokens` correspond.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`UserDict`: A dictionary composed of the fields as defined above:

            - **next_beam_scores** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Updated
              scores of all non-finished beams.
            - **next_beam_tokens** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Next tokens
              to be added to the non-finished beam_hypotheses.
            - **next_beam_indices** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Beam indices
              indicating to which beam the next tokens shall be added.

"""

FINALIZE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        final_beam_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The final scores of all non-finished beams.
        final_beam_tokens (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The last tokens to be added to the non-finished beam_hypotheses.
        final_beam_indices (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The beam indices indicating to which beam the :obj:`final_beam_tokens` shall be added.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
        sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
        batches finished early due to the :obj:`eos_token_id`.

"""


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    def process(self, input_ids: torch.LongTensor,
                next_scores: torch.FloatTensor, next_tokens: torch.LongTensor,
                next_indices: torch.LongTensor,
                **kwargs) -> Tuple[torch.Tensor]:
        raise NotImplementedError('This is an abstract method.')

    @abstractmethod
    def finalize(self, input_ids: torch.LongTensor,
                 next_scores: torch.FloatTensor, next_tokens: torch.LongTensor,
                 next_indices: torch.LongTensor, **kwargs) -> torch.LongTensor:
        raise NotImplementedError('This is an abstract method.')


class BeamSearchScorer(BeamScorer):
    r"""
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            ) for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)],
                                  dtype=torch.bool,
                                  device=self.device)

        # if not isinstance(num_beams, int) or num_beams <= 1:
        #     raise ValueError(
        #     )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(self,
                input_ids: torch.LongTensor,
                next_scores: torch.FloatTensor,
                next_tokens: torch.LongTensor,
                next_indices: torch.LongTensor,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                mems=None) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.num_beams)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        device = next_scores.device
        next_beam_scores = torch.zeros((batch_size, self.num_beams),
                                       dtype=next_scores.dtype,
                                       device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams),
                                       dtype=next_tokens.dtype,
                                       device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams),
                                        dtype=next_indices.dtype,
                                        device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), 'Batch can only be done if at least {} beams have been generated'.format(
                    self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score,
                                  next_index) in enumerate(
                                      zip(next_tokens[batch_idx],
                                          next_scores[batch_idx],
                                          next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item()
                                                   in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        mems=[mem[[next_index.item()]]
                              for mem in mems] if mems else None)
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            if beam_idx < self.num_beams:
                raise ValueError(
                    f'At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected.'  # noqa
                )  # noqa

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len)

        return UserDict({
            'next_beam_scores': next_beam_scores.view(-1),
            'next_beam_tokens': next_beam_tokens.view(-1),
            'next_beam_indices': next_beam_indices.view(-1),
        })

    def finalize(self,
                 input_ids: torch.LongTensor,
                 final_beam_scores: torch.FloatTensor,
                 final_beam_tokens: torch.LongTensor,
                 final_beam_indices: torch.LongTensor,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 mems=None) -> Tuple[torch.LongTensor, List[torch.Tensor]]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(
                    final_tokens,
                    final_score,
                    mems=[mem[[batch_beam_idx]]
                          for mem in mems] if mems else None)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp, mems = sorted_hyps.pop()[1:]
                sent_lengths[self.num_beam_hyps_to_keep * i
                             + j] = len(best_hyp)
                best.append((best_hyp, mems))

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item(), self.max_length)
        decoded: torch.LongTensor = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, '`pad_token_id` has to be defined'
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        mems = []
        for i, (hypo, mem) in enumerate(best):
            decoded[i, :sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id
            mems.append(mem)
        mems = [
            torch.cat([mem[i] for mem in mems], dim=0)
            for i in range(len(mems[0]))
        ] if mems and mems[0] else None
        return decoded, mems


class BeamHypotheses:

    def __init__(self, num_beams: int, max_length: int, length_penalty: float,
                 early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, mems=None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (max(hyp.shape[-1], 1)**self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, mems))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([
                    (s, idx) for idx, (s, _, _) in enumerate(self.beams)
                ])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class LogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.'
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsWarper` to subsequently process a :obj:`scores` input tensor. This class inherits from
    list and adds a specific `__call__` method to apply each :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsProcessor` to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(
                f'`min_length` has to be a positive integer, but is {min_length}'
            )

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(
                f'`eos_token_id` has to be a positive integer, but is {eos_token_id}'
            )

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float('inf')
        return scores


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces no repetition of n-grams. See `Fairseq
    <https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345>`__.

    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f'`ngram_size` has to be a strictly positive integer, but is {ngram_size}'
            )
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = self._calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float('inf')

        return scores

    def _calc_banned_ngram_tokens(self, prev_input_ids: torch.Tensor,
                                  num_hypos: int,
                                  cur_len: int) -> List[Iterable[int]]:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < self.ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:]
                               for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                    prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx,
                                             start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [
            _get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)
        ]
        return banned_tokens

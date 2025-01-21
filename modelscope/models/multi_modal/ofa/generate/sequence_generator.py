# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modelscope.models.multi_modal.ofa.generate import search
from modelscope.models.multi_modal.ofa.generate.ngram_repeat_block import \
    NGramRepeatBlock


def _expand_mask(mask: torch.Tensor,
                 dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    r"""
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)
    return expanded_mask.masked_fill(expanded_mask.bool(),
                                     torch.finfo(dtype).min)


class SequenceGenerator(nn.Module):

    def __init__(self,
                 tokenizer,
                 beam_size=1,
                 max_len_a=0,
                 max_len_b=200,
                 max_len=0,
                 min_len=1,
                 normalize_scores=True,
                 len_penalty=1.0,
                 unk_penalty=0.0,
                 temperature=1.0,
                 match_source_len=False,
                 no_repeat_ngram_size=0,
                 search_strategy=None,
                 eos=None,
                 symbols_to_strip_from_output=None,
                 lm_model=None,
                 lm_weight=1.0,
                 constraint_trie=None,
                 constraint_range=None,
                 gen_code=False,
                 gen_box=False,
                 ignore_eos=False,
                 zero_shot=False):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        self.gen_code = gen_code
        self.gen_box = gen_box
        self.ignore_eos = ignore_eos
        self.tokenizer = tokenizer
        self.tgt_dict = {
            value: key
            for key, value in tokenizer.get_vocab().items()
        }
        added = {
            value: key
            for key, value in tokenizer.get_added_vocab().items()
        }
        self.tgt_dict.update(added)
        self.pad = tokenizer.pad_token_id
        self.unk = tokenizer.unk_token_id
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos}) if
            symbols_to_strip_from_output is not None else {self.bos, self.eos})
        self.vocab_size = len(self.tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.zero_shot = zero_shot

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, '--temperature must be greater than 0'

        self.search = (
            search.BeamSearch(self.tokenizer)
            if search_strategy is None else search_strategy)
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, 'needs_src_lengths')
            and self.search.needs_src_lengths)

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.constraint_trie = constraint_trie

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]],
                 **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(models, sample, **kwargs)

    def _generate(
        self,
        models,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        model = EnsembleModel(models)
        incremental_states = torch.jit.annotate(
            List[Tuple[Tuple[torch.Tensor]]],
            [
                torch.jit.annotate(Tuple[Tuple[torch.Tensor]], {})
                for i in range(model.models_size)
            ],
        )
        net_input = sample['net_input']

        if 'src_tokens' in net_input:
            src_tokens = net_input['src_tokens']
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = ((src_tokens.ne(self.eos)
                            & src_tokens.ne(self.pad)).long().sum(dim=1))
        elif 'input_ids' in net_input:
            src_tokens = net_input['input_ids']
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = ((src_tokens.ne(self.eos)
                            & src_tokens.ne(self.pad)).long().sum(dim=1))
        elif 'source' in net_input:
            src_tokens = net_input['source']
            src_lengths = (
                net_input['padding_mask'].size(-1)
                - net_input['padding_mask'].sum(-1)
                if net_input['padding_mask'] is not None else torch.tensor(
                    src_tokens.size(-1)).to(src_tokens))
        elif 'features' in net_input:
            src_tokens = net_input['features']
            src_lengths = (
                net_input['padding_mask'].size(-1)
                - net_input['padding_mask'].sum(-1)
                if net_input['padding_mask'] is not None else torch.tensor(
                    src_tokens.size(-1)).to(src_tokens))
        elif 'fbank' in net_input:
            src_tokens = net_input['fbank']
            src_lengths = net_input['fbank_length']
        else:
            raise Exception(
                'expected src_tokens or source in net input. input keys: '
                + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = int(self.max_len_a * src_len + self.max_len_b)
        assert (
            self.min_len <= max_len
        ), 'min_len cannot be larger than max_len, please adjust these!'
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function(
                'EnsembleModel: forward_encoder'):
            encoder_outs = model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (torch.zeros(bsz * beam_size,
                              max_len + 1).to(src_tokens).float()
                  )  # +1 for eos; pad is never chosen for scoring
        tokens = (torch.zeros(bsz * beam_size,
                              max_len + 2).to(src_tokens).long().fill_(
                                  self.pad))  # +2 for eos and pad
        tokens[:, 0] = self.bos
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
                           )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [
                torch.jit.annotate(List[Dict[str, Tensor]], [])
                for i in range(bsz)
            ],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = ((torch.arange(0, bsz)
                         * beam_size).unsqueeze(1).type_as(tokens).to(
                             src_tokens.device))
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(
            src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if 'id' in sample and isinstance(sample['id'], Tensor):
            original_batch_idxs = sample['id']
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(
                        batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size)
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                model.reorder_incremental_state(incremental_states,
                                                reorder_state)
                encoder_outs = model.reorder_encoder_out(
                    encoder_outs, reorder_state)

            with torch.autograd.profiler.record_function(
                    'EnsembleModel: forward_decoder'):
                lprobs, avg_attn_scores = model.forward_decoder(
                    tokens[:, :step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                    constraint_trie=self.constraint_trie,
                    constraint_start=self.constraint_start,
                    constraint_end=self.constraint_end,
                    gen_code=self.gen_code,
                    zero_shot=self.zero_shot,
                    prefix_tokens=prefix_tokens)

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, :step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None)
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            # handle prefix tokens (possibly with different lengths)
            if (prefix_tokens is not None and step < prefix_tokens.size(1)
                    and step < max_len):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            if (self.gen_code or self.gen_box) and step < max_len:
                lprobs[:, :4] = -math.inf
            if self.gen_box:
                lprobs[:, -1] = -math.inf
                if (step + 1) % 5 == 0:
                    lprobs[:, self.constraint_start:59457] = -math.inf
                else:
                    lprobs[:, 59457:] = -math.inf

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
                if self.ignore_eos:
                    lprobs[:, self.eos] = 1

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(bsz * beam_size,
                                       avg_attn_scores.size(1),
                                       max_len + 2).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                # process prefix_tokens
                p_toks_len = prefix_tokens.ne(self.pad).sum(
                    dim=1) if prefix_tokens is not None else None
                if p_toks_len is not None:
                    p_toks_len_beam = p_toks_len.unsqueeze(-1).repeat(
                        1, beam_size).view(-1)
                    no_repeat_ngram_size = self.repeat_ngram_blocker.no_repeat_ngram_size
                    out_prefix = p_toks_len_beam < (
                        step + no_repeat_ngram_size - 1)
                else:
                    out_prefix = torch.ones(bsz * beam_size).bool()
                ngram_blocker_tokens = tokens[out_prefix]
                ngram_blocker_lprobs = lprobs[out_prefix]
                ngram_blocker_bsz = torch.div(
                    out_prefix.sum(), beam_size, rounding_mode='trunc')

                lprobs[out_prefix] = self.repeat_ngram_blocker(
                    tokens=ngram_blocker_tokens,
                    lprobs=ngram_blocker_lprobs,
                    bsz=ngram_blocker_bsz,
                    beam_size=beam_size,
                    step=step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, :step + 1],
                original_batch_idxs,
            )
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(
                eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f'{step} < {max_len}'

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~(  # noqa
                (~cands_to_ignore) & (~eos_mask[:, :beam_size]))  # noqa
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False)

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, :step + 1] = torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx)
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos)
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx)
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos)

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, :step + 2] = torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx)

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem['score'].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [
                finalized[sent][ssi] for ssi in sorted_scores_indices
            ]
            finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]],
                                                 finalized[sent])
        return finalized

    def _prefix_tokens(self, step: int, lprobs, scores, tokens, prefix_tokens,
                       beam_size: int):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(
            1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        if self.constraint_trie is None:
            lprobs[prefix_mask] = torch.min(prefix_lprobs) - 1
        else:
            lprobs[prefix_mask] = -math.inf
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1),
            prefix_lprobs[prefix_mask])
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size,
                                               tokens.size(-1))[:, 0,
                                                                1:step + 1]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim,
                                               beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim,
                                               beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim,
                                               beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(
            0, bbsz_idx)[:, 1:step + 2]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1:step
                                           + 2] if attn is not None else None)

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1)**self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode='floor')
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf),
                                     eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append({
                    'tokens':
                    tokens_clone[i],
                    'score':
                    eos_scores[i],
                    'attention':
                    hypo_attn,  # src_len x tgt_len
                    'alignment':
                    torch.empty(0),
                    'positional_scores':
                    pos_scores[i],
                })

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                    step, unique_unfin_idx, max_len, len(
                        finalized[unique_sent]), beam_size):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        # self.has_incremental: bool = False
        # if all(
        #     hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
        #     for m in models
        # ):
        #     self.has_incremental = True

        self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, 'encoder')

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([
            m.max_decoder_positions()
            for m in self.models if hasattr(m, 'max_decoder_positions')
        ] + [sys.maxsize])  #

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        encoder_input = {
            k: v
            for k, v in net_input.items() if k != 'decoder_input_ids'
        }
        encoder_input['output_hidden_states'] = True
        return [
            model.encoder.forward(**encoder_input) for model in self.models
        ]

    @torch.jit.export
    def forward_decoder(self,
                        tokens,
                        encoder_outs: List[Dict[str, List[Tensor]]],
                        incremental_states: List[Optional[torch.Tensor]],
                        temperature: float = 1.0,
                        constraint_trie=None,
                        constraint_start=None,
                        constraint_end=None,
                        gen_code=False,
                        zero_shot=False,
                        prefix_tokens=None):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        code_mask = (tokens.new_ones(tokens.size(0)) * gen_code).bool()

        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
                encoder_hidden_states = encoder_out.last_hidden_state
                encoder_attention_mask = _expand_mask(
                    encoder_out.padding_mask, encoder_hidden_states.dtype,
                    tokens.shape[-1])
                src_pos_embed = encoder_out.position_embedding

                # if tokens.eq(self.single_model.config.pad_token_id).any():
                attention_mask = tokens.eq(self.single_model.padding_idx)

            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    code_masks=code_mask,
                    src_pos_embed=src_pos_embed,
                    past_key_values=incremental_states[i],
                    use_cache=True,
                    output_attentions=True)
            else:
                if hasattr(model, 'decoder'):
                    # decoder_out = model.decoder.forward(tokens, code_masks=code_mask, encoder_out=encoder_out)
                    decoder_out = model.decoder.forward(
                        input_ids=tokens,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        code_masks=code_mask,
                        src_pos_embed=src_pos_embed)
                else:
                    decoder_out = model.forward(tokens)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)

            if 'cross_attentions' in decoder_out:
                attn = decoder_out['cross_attentions'][-1].transpose(1, 0)
                attn = attn.mean(dim=0)  # (B, tgt_len, src_len)
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else attn,
            )

            beam_size = decoder_out_tuple[0].size(0) // prefix_tokens.size(
                0) if prefix_tokens is not None else 0
            if constraint_trie is not None and not zero_shot:
                assert constraint_start is None and constraint_end is None
                constraint_masks = decoder_out_tuple[0].new_zeros(
                    decoder_out_tuple[0].size()).bool()
                constraint_prefix_tokens = tokens.tolist()
                for token_index, constraint_prefix_token in enumerate(
                        constraint_prefix_tokens):
                    prefix_len = prefix_tokens[token_index // beam_size].ne(
                        1).sum().item() if prefix_tokens is not None else 0
                    if len(constraint_prefix_token) > prefix_len:
                        constraint_prefix_token = [
                            0
                        ] + constraint_prefix_token[prefix_len + 1:]
                        constraint_nodes = constraint_trie.get_next_layer(
                            constraint_prefix_token)
                        constraint_masks[token_index][:,
                                                      constraint_nodes] = True
                    else:
                        constraint_masks[token_index] = True
                decoder_out_tuple[0].masked_fill_(~constraint_masks, -math.inf)
            if constraint_start is not None and constraint_end is not None and not zero_shot:
                assert constraint_trie is None
                decoder_out_tuple[0][:, :, 4:constraint_start] = -math.inf
                decoder_out_tuple[0][:, :, constraint_end:] = -math.inf

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None)
            if constraint_trie is not None and zero_shot:
                assert constraint_start is None and constraint_end is None
                constraint_masks = decoder_out_tuple[0].new_zeros(
                    decoder_out_tuple[0].size()).bool()
                constraint_prefix_tokens = tokens.tolist()
                for token_index, constraint_prefix_token in enumerate(
                        constraint_prefix_tokens):
                    constraint_nodes = constraint_trie.get_next_layer(
                        constraint_prefix_token)
                    constraint_masks[token_index][:, constraint_nodes] = True
                probs.masked_fill_(~constraint_masks, -math.inf)
            if constraint_start is not None and constraint_end is not None and zero_shot:
                assert constraint_trie is None
                probs[:, :, 4:constraint_start] = -math.inf
                probs[:, :, constraint_end:] = -math.inf
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(
            torch.stack(log_probs, dim=0), dim=0) - math.log(self.models_size)

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(self,
                            encoder_outs: Optional[List[Dict[str,
                                                             List[Tensor]]]],
                            new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order))
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Optional[torch.Tensor]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order)

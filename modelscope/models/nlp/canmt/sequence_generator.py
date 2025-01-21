# Part of the implementation is borrowed and modified from FAIRSEQ,
# publicly available at https://github.com/facebookresearch/fairseq
# Copyright 2022-2023 The Alibaba MT Team Authors. All rights reserved.
import math
import sys
from typing import Dict, List, Optional

import numpy
import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.ngram_repeat_block import NGramRepeatBlock
from torch import Tensor


def label_smoothed_nll_loss(lprobs,
                            target,
                            epsilon,
                            ignore_index=None,
                            reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if target.dtype != torch.int64:
        target = target.type(torch.int64)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class SequenceGenerator(nn.Module):

    def __init__(
        self,
        model,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=200,
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
        recon_force_decoding=True,
        trans_force_decoding=False,
    ):
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
        self.model = model
        self.recon_force_decoding = recon_force_decoding
        self.trans_force_decoding = trans_force_decoding
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None else {self.eos})
        self.vocab_size = len(tgt_dict)
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
        self.eps = 0.1

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, '--temperature must be greater than 0'

        self.search = (
            search.BeamSearch(tgt_dict)
            if search_strategy is None else search_strategy)
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, 'needs_src_lengths')
            and self.search.needs_src_lengths)

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

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

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self,
                             data_itr,
                             beam_size=None,
                             cuda=False,
                             timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v
                for k, v in input.items() if k != 'prev_output_tokens'
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                # remove padding
                src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s['target'].data[i, :], self.pad)
                    if s['target'] is not None else None)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, sample: Dict[str, Dict[str, Tensor]],
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
        from torch import tensor
        finalized = self._generate(sample, **kwargs)
        tokens_list = []
        decoder_list = []
        for i in range(len(finalized)):
            sent = finalized[i][0]
            tokens = sent['tokens']
            tokens_list.append(tokens)
            decoder_out = sent['decoder_out']
            decoder_list.append(decoder_out)
        # padding tokens
        size = max(v.size(0) for v in tokens_list)
        batch_size = len(tokens_list)

        for i in range(len(tokens_list)):
            tokens_list[i] = tokens_list[i].roll(1, 0)
            decoder_list[i] = decoder_list[i].roll(1, 0)
        res = tokens_list[0].new(batch_size, size).fill_(self.pad)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(tokens_list):
            copy_tensor(v, res[i][:len(v)] if True else res[i][:len(v)])
        tgt_tokens = res
        decoder_padding_mask = tgt_tokens.eq(self.pad)

        # generate for src reconstruction
        decoder_out_re = self.model.decoder(
            tgt_tokens,
            encoder_out=None,
            full_context_alignment=True,
        )
        decoder_out_re = decoder_out_re[1]['last_layer']
        decoder_outs = dict()
        decoder_outs['encoder_out'] = [decoder_out_re]
        decoder_outs['encoder_padding_mask'] = [decoder_padding_mask]
        decoder_outs['src_tokens'] = [tgt_tokens]
        decoder_outs['encoder_embedding'] = []
        decoder_outs['encoder_states'] = []
        decoder_outs['src_lengths'] = []

        scores = self._forward_src(decoder_outs, sample)
        return finalized, scores

    def _forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
    ):
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        prev_output_tokens = net_input['prev_output_tokens']
        encoder_outs = self.model.forward_encoder(net_input)
        final_encoder_out = encoder_outs['encoder_out'][0].transpose(0, 1)
        final_encoder_embedding = encoder_outs['encoder_embedding'][0]
        self.model.has_incremental = False
        lprobs, avg_attn_scores, decoder_outs = self.model.forward_decoder(
            prev_output_tokens, encoder_outs, incremental_states=None)
        decoder_outs = decoder_outs.transpose(0, 1)
        self.model.has_incremental = True
        batch_size = src_tokens.size(0)
        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [
                torch.jit.annotate(List[Dict[str, Tensor]], [])
                for i in range(batch_size)
            ],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step
        for i in range(batch_size):
            eos_idx = numpy.where(sample['target'][i].cpu() == self.eos)[0][0]
            finalized[i].append({
                'tokens':
                sample['target'][i][:eos_idx + 1],
                'decoder_out':
                decoder_outs[i][:eos_idx + 1],
                'final_encoder_embedding':
                final_encoder_embedding[i],
                'final_encoder_out':
                final_encoder_out[i],
            })
        return finalized

    def _forward_src(
        self,
        encoder_outs,
        sample: Dict[str, Dict[str, Tensor]],
    ):
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        prev_output_tokens = net_input['prev_src_tokens']
        self.model.has_incremental = False
        lprobs, avg_attn_scores, _, decoder_outs = self.model.forward_decoder_src(
            prev_output_tokens, encoder_outs, incremental_states=None)
        logits = decoder_outs[0]
        lprobs = utils.log_softmax(logits, dim=-1)
        sources = net_input['sources'].view(-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        scores, _ = label_smoothed_nll_loss(
            lprobs,
            sources,
            epsilon=self.eps,
            ignore_index=self.pad,
            reduce=False)
        batch_size = src_tokens.shape[0]
        scores = scores.reshape(batch_size, -1)
        scores = torch.cat((scores.sum(axis=-1, keepdim=True)
                            / src_lengths.reshape(batch_size, 1), scores),
                           axis=-1)
        return scores

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )

        net_input = sample['net_input']
        if 'src_tokens' in net_input:
            src_tokens = net_input['src_tokens']
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = ((src_tokens.ne(self.eos)
                            & src_tokens.ne(self.pad)).long().sum(dim=1))
        else:
            raise Exception(
                'expected src_tokens or source in net input. input keys: '
                + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), 'min_len cannot be larger than max_len, please adjust these!'
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        final_encoder_out = encoder_outs['encoder_out'][0].transpose(0, 1)
        final_encoder_embedding = encoder_outs['encoder_embedding'][0]

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)

        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (torch.zeros(bsz * beam_size,
                              max_len + 1).to(src_tokens).float()
                  )  # +1 for eos; pad is never chosen for scoring
        tokens = (torch.zeros(bsz * beam_size,
                              max_len + 2).to(src_tokens).long().fill_(
                                  self.pad))  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

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

                self.model.reorder_incremental_state(incremental_states,
                                                     reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state)

            lprobs, avg_attn_scores, decoder_out_word = self.model.forward_decoder(
                tokens[:, :step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
            )
            # (length, batch*beam, hidden_size) - >(batch*beam, length, hidden_size)
            decoder_out_word = decoder_out_word.transpose(0, 1)
            if step == 0:
                decoder_out_tensor = decoder_out_word
            else:
                decoder_out_tensor = torch.cat(
                    [decoder_out_tensor, decoder_out_word], dim=1)

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (prefix_tokens is not None and step < prefix_tokens.size(1)
                    and step < max_len):

                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

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
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz,
                                                   beam_size, step)

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
                    decoder_out_tensor,
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
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
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
                decoder_out_tensor = decoder_out_tensor.contiguous().view(
                    bsz, -1)[batch_idxs].view(new_bsz * beam_size,
                                              decoder_out_tensor.size(1), -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            eos_mask[:, :beam_size] = ~((~cands_to_ignore)
                                        & (~eos_mask[:, :beam_size]))
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
            finalized[sent][0][
                'final_encoder_embedding'] = final_encoder_embedding[sent]
            finalized[sent][0]['final_encoder_out'] = final_encoder_out[sent]
        return finalized

    def _prefix_tokens(self, step: int, lprobs, scores, tokens, prefix_tokens,
                       beam_size: int):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(
            1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
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
        decoder_out=None,
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
        if decoder_out is not None:
            decoder_out_clone = decoder_out.index_select(0, bbsz_idx)

        tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1:step + 2]
        # skip the first index, which is EOS

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

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + '_' + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent].append({
                    'tokens':
                    tokens_clone[i],
                    'score':
                    score,
                    'attention':
                    hypo_attn,  # src_len x tgt_len
                    'alignment':
                    torch.empty(0),
                    'positional_scores':
                    pos_scores[i],
                    'decoder_out':
                    decoder_out_clone[i] if decoder_out is not None else [],
                })

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split('_')[0]))
            unfin_idx: int = int(float(seen.split('_')[1]))

            if not finished[sent] and self.is_finished(
                    step, unfin_idx, max_len, len(finalized[sent]), beam_size):
                finished[sent] = True
                newly_finished.append(unfin_idx)

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

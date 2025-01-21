# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import torch
import torch.nn.functional as F


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = None

    translator = TextGenerator(
        args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class TextGenerator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab=None,
                 symbols=None,
                 global_scorer=None,
                 logger=None,
                 dump_beam=''):
        self.alpha = 0.6

        self.logger = logger
        self.cuda = (torch.cuda.device_count() > 0)

        self.args = args
        self.model = model

        self.vocab = vocab
        self.symbols = symbols
        self.start_token = 101  # ['[PAD]']
        self.end_token = 102  # ['[PAD]']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ''
        self.beam_accum = None

        if self.beam_trace:
            self.beam_accum = {
                'predicted_ids': [],
                'beam_parent_ids': [],
                'scores': [],
                'log_probs': []
            }

    def _build_target_tokens(self, pred):
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def translate_batch(self, encoder_inputs, do_sample=False, out_size=1):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        if do_sample:
            return self._fast_translate_batch(
                encoder_inputs,
                self.max_length,
                min_length=self.min_length,
                do_sample=do_sample,
                out_size=out_size)
        else:
            with torch.no_grad():
                return self._fast_translate_batch(
                    encoder_inputs,
                    self.max_length,
                    min_length=self.min_length,
                    do_sample=do_sample,
                    out_size=out_size)

    def translate_batch_scst(self,
                             encoder_inputs,
                             do_sample=False,
                             out_size=1):
        return self._fast_translate_batch(
            encoder_inputs,
            self.max_length,
            min_length=self.min_length,
            do_sample=do_sample,
            out_size=out_size)

    def _fast_translate_batch(self,
                              encoder_inputs,
                              max_length,
                              min_length=0,
                              do_sample=False,
                              out_size=1):

        assert not self.dump_beam
        if do_sample:
            beam_size = 1
        else:
            beam_size = self.beam_size
        if len(encoder_inputs) == 3:
            src_features, padding_mask, input_ids = encoder_inputs
        elif len(encoder_inputs) == 2:
            src_features, padding_mask = encoder_inputs
            input_ids = None

        device = src_features.device

        # Tile states and memory beam_size times.
        batch_size = src_features.size(0)
        src_features = tile(src_features, beam_size, dim=0)
        attention_mask = tile(padding_mask, beam_size, dim=0)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        if input_ids is not None:
            alive_seq = tile(input_ids, beam_size, dim=0)
        else:
            alive_seq = torch.full([batch_size * beam_size, 1],
                                   self.start_token,
                                   dtype=torch.long,
                                   device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor(
                [0.0] + [float('-inf')] * (beam_size - 1),
                device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results['predictions'] = [[] for _ in range(batch_size)]  # noqa: F812
        results['scores'] = [[] for _ in range(batch_size)]  # noqa: F812
        results['gold_score'] = [0] * batch_size
        results['batch'] = []

        for step in range(max_length):
            dec_feat_seq = self.model(
                alive_seq,
                encoder_hidden_states=src_features,
                encoder_attention_mask=attention_mask,
                return_dict=True,
                reduction='none')

            dec_feat_seq = dec_feat_seq.logits[:, -1, :]
            vocab_size = dec_feat_seq.size(-1)
            log_probs = torch.log(
                torch.softmax(dec_feat_seq.view(-1, vocab_size), dim=-1))
            if step < min_length:
                log_probs[:, self.end_token] = -1e20
            alpha = self.alpha
            if do_sample:
                length_penalty = 1.0
            else:
                length_penalty = ((5.0 + (step + 1)) / 6.0)**alpha

            if do_sample:
                _scores = log_probs / self.args.temperature
                _scores = top_k_top_p_filtering(
                    _scores,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    min_tokens_to_keep=1
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam
                # (so we have some spare tokens and match output of greedy beam search)
                topk_ids = torch.multinomial(
                    F.softmax(_scores, dim=-1),
                    num_samples=1)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(
                    _scores, dim=1)  # (batch_size * num_beams, vocab_size)

                _scores += topk_log_probs.view(-1).unsqueeze(1)
                topk_scores = torch.gather(
                    _scores, -1, topk_ids)  # (batch_size * num_beams, 2)
                # log_probs +=   # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                topk_ids = topk_ids.view(
                    -1, beam_size)  # (batch_size, 2 * num_beams)
                topk_scores = topk_scores.view(
                    -1, beam_size)  # (batch_size, 2 * num_beams)
            else:
                log_probs += topk_log_probs.view(-1).unsqueeze(1)
                curr_scores = log_probs / length_penalty

                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
                topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            # topk_beam_index = topk_ids.div(vocab_size)
            topk_beam_index = torch.div(
                topk_ids, vocab_size, rounding_mode='floor')
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat([
                alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)
            ], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)  # self.end_token)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)  # self.end_token)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)  # self.end_token)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 0:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)

                        for each in best_hyp[:beam_size]:
                            score, pred = each
                            results['scores'][b].append(score)
                            results['predictions'][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            attention_mask = attention_mask.index_select(0, select_indices)
        pred_ids = []
        scores = []
        # print (pred_ids, scores)
        for each in results['scores']:
            scores.append(each[:out_size])
        for each in results['predictions']:
            pred_ids.append(each[:out_size])
        return pred_ids, scores

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past)
            outputs = self(**model_inputs)
            if cur_len == 1:
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1
            assert outputs[0].shape[1] == token_len

            next_token_logits = outputs[0][:, next_token_idx, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied
                        # to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[
                                i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[
                                i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(
                    F.softmax(next_token_logits, dim=-1),
                    num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(
                next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1,
                                   next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (
                1 - cur_unfinished)
            input_ids = torch.cat(
                [input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(
                    tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(
                cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size,
                                        pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)


def top_k_top_p_filtering(logits,
                          top_k=10,
                          top_p=1.0,
                          filter_value=-float('Inf'),
                          min_tokens_to_keep=1):

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep),
                    logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents, attn, pred_scores,
                 tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += 'PRED SCORE: {:.4f}\n'.format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ('GOLD SCORE: {:.4f}\n'.format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += '[{:.4f}] {}\n'.format(score, sent)

        return output


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

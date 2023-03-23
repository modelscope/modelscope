# Copyright (c) 2022 Zhipu.AI
import numpy as np
import torch
import torch.nn.functional as F
from SwissArmyTransformer.generation.sampling_strategies.base_strategy import \
    top_k_logits


class BaseStrategy:

    def __init__(self,
                 batch_size,
                 invalid_slices=[],
                 temperature=1.,
                 top_k=200,
                 eps=1e-4,
                 top_p=0.0,
                 end_tokens=None):
        self.batch_size = batch_size
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = np.zeros(self.batch_size, dtype=bool)

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems, temperature=None):
        logits = logits.view(-1, logits.size(-1))
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504

        logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(
            logits.float(),
            dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        for i in range(self.batch_size):
            if i >= batch_size:
                self._is_done[i] = True
            elif self._is_done[i]:
                pred[i] = -1
            elif pred[i].item() in self.end_tokens:
                self._is_done[i] = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[:-1] + (1, ))),
                           dim=-1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = np.zeros(self.batch_size, dtype=bool)
        return tokens, mems


class BeamSearchStrategy:

    def __init__(
        self,
        batch_size,
        num_beams,
        length_penalty=1.0,
        consider_end=False,
        end_tokens=[],
        invalid_slices=[],
        no_repeat_ngram_size=0,
        min_gen_length=0,
        deterministic=False,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_gen_length = min_gen_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.deterministic = deterministic
        self._init_cache()

    def _init_cache(self):
        self.end_beams = [[] for _ in range(self.batch_size)
                          ]  # list of LongTensors
        self.end_beams_penalized_scores = [[] for _ in range(self.batch_size)
                                           ]  # list of LongTensors
        self.cached_beam_scores = 0  # [batch_size]
        self.cached_beam_ngram_bans = [[{} for _ in range(self.num_beams)]
                                       for _ in range(self.batch_size)]
        self.length_generated = 0
        self._is_done = np.zeros(self.batch_size, dtype=bool)

    def _add_end_beams(self, score, beam, batch_idx):
        score = score / ((5.0 + len(beam))
                         / 6)**self.length_penalty  # Magic number for OpenNMT
        for i in range(len(self.end_beams[batch_idx]), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[batch_idx][
                    i - 1]:
                break
        self.end_beams[batch_idx].insert(i, beam)
        self.end_beams_penalized_scores[batch_idx].insert(i, score)

        self.end_beams[batch_idx] = self.end_beams[batch_idx][:self.num_beams]
        self.end_beams_penalized_scores[
            batch_idx] = self.end_beams_penalized_scores[batch_idx][:self.
                                                                    num_beams]

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems):
        batch_size, num_beams, vocab_size = logits.shape
        seq_len = tokens.shape[-1]
        logits = logits.float()
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if self.min_gen_length > self.length_generated:
            for end_token in self.end_tokens:
                logits[..., end_token] = -65504
        if self.ngram > 0 and seq_len > self.ngram:
            for batch_idx in range(batch_size):
                for i in range(num_beams):
                    ngram_prefix = tokens[batch_idx, i,
                                          -(self.ngram
                                            - 1):].tolist()  # TODO ngram=1
                    for banned_index in self.cached_beam_ngram_bans[batch_idx][
                            i].get(tuple(ngram_prefix), []):
                        logits[batch_idx, i, banned_index] = -65504

        next_token_scores = F.log_softmax(
            logits, dim=-1)  # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        if isinstance(prev_scores, torch.Tensor):
            prev_scores = prev_scores[..., None].expand_as(next_token_scores)
        next_token_scores = next_token_scores + prev_scores

        next_token_scores = next_token_scores.view(batch_size,
                                                   num_beams * vocab_size)

        probs = F.softmax(next_token_scores, dim=-1)
        if num_beams < self.num_beams:  # First token
            probs = probs[..., :vocab_size]
        if self.deterministic:
            next_tokens = torch.topk(
                probs, k=(max(1, len(self.end_tokens)) + 1)
                * self.num_beams).indices  # [2*nb]
        else:
            next_tokens = torch.multinomial(
                probs,
                num_samples=(max(1, len(self.end_tokens)) + 1)
                * self.num_beams)  # [2*nb]
        next_token_scores = next_token_scores[
            torch.arange(batch_size).unsqueeze(1), next_tokens]
        next_token_scores, _indices = torch.sort(
            next_token_scores, descending=True, dim=1)
        next_tokens = next_tokens[torch.arange(batch_size).unsqueeze(1),
                                  _indices]

        next_indices = torch.div(
            next_tokens, vocab_size, rounding_mode='trunc')
        next_tokens = next_tokens % vocab_size

        # select out end beams or continue beams
        beam_continue_batch, score_continue_batch, mems_continue_batch = [], [], []
        for batch_idx in range(batch_size):
            beam_continue = []
            scores_continue = []
            bans_continue = []
            mems_contiue = []
            for i in range(len(next_tokens[batch_idx])):
                beam = torch.cat(
                    (tokens[batch_idx, next_indices[batch_idx,
                                                    i]], next_tokens[batch_idx,
                                                                     i:i + 1]))
                if not self._is_done[batch_idx] and int(
                        next_tokens[batch_idx, i]) in self.end_tokens:
                    self._add_end_beams(next_token_scores[batch_idx, i], beam,
                                        batch_idx)
                elif len(beam_continue) < self.num_beams:
                    beam_continue.append(beam)
                    mems_contiue.append(mems[:, batch_idx,
                                             next_indices[batch_idx, i]])
                    # update caches
                    scores_continue.append(next_token_scores[batch_idx, i])
                    if self.ngram > 0:
                        bans = self.cached_beam_ngram_bans[batch_idx][
                            next_indices[batch_idx, i]].copy()
                        # TODO ngram=1
                        ngram_prefix = tuple(
                            tokens[batch_idx, next_indices[batch_idx, i],
                                   -(self.ngram - 1):].tolist())
                        bans[ngram_prefix] = bans.get(
                            ngram_prefix, tuple()) + (next_tokens[batch_idx,
                                                                  i], )
                        bans_continue.append(bans)
                else:
                    break
            beam_continue_batch.append(torch.stack(beam_continue))
            mems_continue_batch.append(torch.stack(mems_contiue, dim=1))
            score_continue_batch.append(scores_continue)
            self.cached_beam_ngram_bans[batch_idx] = bans_continue
        tokens = torch.stack(beam_continue_batch)
        mems = torch.stack(mems_continue_batch, dim=1)
        self.cached_beam_scores = torch.tensor(
            score_continue_batch, device=logits.device)
        self.length_generated += 1
        for batch_idx in range(self.batch_size):
            if batch_idx >= batch_size:
                self._is_done[batch_idx] = True
            elif (len(self.end_beams[batch_idx]) == self.num_beams
                  and self.end_beams_penalized_scores[batch_idx][-1] >=  # noqa
                  self.cached_beam_scores[batch_idx].max() /  # noqa
                  ((5.0 + (seq_len + 1)) / 6)**self.length_penalty):  # noqa
                self._is_done[batch_idx] = True

        return tokens, mems

    def finalize(self, tokens, mems):
        if self.consider_end:
            batch_size, num_beams = tokens.shape[:2]
            for batch_idx in range(batch_size):
                if not self._is_done[batch_idx]:
                    for i in range(num_beams):
                        self._add_end_beams(
                            self.cached_beam_scores[batch_idx, i],
                            tokens[batch_idx, i], batch_idx)
            mems = None
            ret = self.end_beams[:batch_size]
        else:
            ret = tokens
        self._init_cache()
        return ret, mems

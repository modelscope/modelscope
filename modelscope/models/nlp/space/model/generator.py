# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import numpy as np
import torch


def repeat(var, times):
    if isinstance(var, list):
        return [repeat(x, times) for x in var]
    elif isinstance(var, dict):
        return {k: repeat(v, times) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        var = var.unsqueeze(1)
        expand_times = [1] * len(var.shape)
        expand_times[1] = times
        dtype = var.dtype
        var = var.float()
        var = var.repeat(*expand_times)
        shape = [var.shape[0] * var.shape[1]] + list(var.shape[2:])
        var = var.reshape(*shape)
        var = torch.tensor(var, dtype=dtype)
        return var
    else:
        return var


def gather(var, idx):
    if isinstance(var, list):
        return [gather(x, idx) for x in var]
    elif isinstance(var, dict):
        return {k: gather(v, idx) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        out = var.index_select(dim=0, index=idx)
        return out
    else:
        return var


class SpaceGenerator(object):
    """ Genrator class. """

    _registry = dict()

    @classmethod
    def register(cls, name):
        SpaceGenerator._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return SpaceGenerator._registry[name]

    @staticmethod
    def create(config, *args, **kwargs):
        """ Create generator. """
        generator_cls = SpaceGenerator.by_name(config.Generator.generator)
        return generator_cls(config, *args, **kwargs)

    def __init__(self, config, reader):
        self.vocab_size = reader.vocab_size
        self.bos_id = reader.bos_id
        self.eos_id = reader.eos_id
        self.unk_id = reader.unk_id
        self.pad_id = reader.pad_id
        self.min_gen_len = config.Generator.min_gen_len
        self.max_gen_len = config.Generator.max_gen_len
        self.use_gpu = config.use_gpu
        if torch.cuda.is_available():
            self.use_gpu = True
        assert 1 <= self.min_gen_len <= self.max_gen_len
        return

    def __call__(self, step_fn, state):
        """Running generation.

        Args:
            step_fn (`function`) : decoding one step
            state(`dict`) : initial state
        """
        raise NotImplementedError


class BeamSearch(SpaceGenerator):
    """ BeamSearch generator. """

    def __init__(self, config, reader):
        super().__init__(config, reader)
        self.beam_size = config.Generator.beam_size
        self.length_average = config.Generator.length_average
        self.length_penalty = config.Generator.length_penalty
        self.ignore_unk = config.Generator.ignore_unk
        return

    def __call__(self,
                 step_fn,
                 state,
                 start_id=None,
                 eos_id=None,
                 max_gen_len=None,
                 prev_input=None):
        """
        Running beam search.

        Args:
            step_fn(`function`) : decoding one step
            state(`dict`) : initial state
        """
        if prev_input is not None:

            if isinstance(prev_input, list):
                length = max(list(map(lambda x: len(x), prev_input)))
                prev_input_numpy = np.full((len(prev_input), length),
                                           self.pad_id)
                for i, x in enumerate(prev_input):
                    prev_input_numpy[i, :len(x)] = x
                prev_input_tensor = torch.from_numpy(prev_input_numpy)
                if self.use_gpu:
                    prev_input_tensor = prev_input_tensor.cuda()

                for i in range(length):
                    state['pred_token'] = prev_input_tensor[:, i].unsqueeze(
                        -1).unsqueeze(-1)
                    if i != 0:
                        state['pred_mask'] = torch.not_equal(
                            state['pred_token'], self.pad_id).float()
                        state['pred_pos'] = state['pred_pos'] + state[
                            'pred_mask'].int()
                    _, state = step_fn(state)
            else:
                assert isinstance(prev_input, torch.Tensor)
                for i, input in enumerate(prev_input):
                    state['pred_token'] = input.expand(1, 1, 1)
                    if i != 0:
                        state['pred_mask'] = torch.not_equal(
                            state['pred_token'], self.pad_id).float()
                        state['pred_pos'] = state['pred_pos'] + 1
                    _, state = step_fn(state)

        batch_size = state['batch_size']
        beam_size = self.beam_size

        # shape: [batch_size, 1]
        pos_index = torch.arange(
            0, batch_size, 1, dtype=torch.int64) * beam_size
        pos_index = pos_index.unsqueeze(1)

        # shape: [batch_size, beam_size, 1]
        if start_id is None:
            start_id = self.bos_id
        if eos_id is None:
            eos_id = self.eos_id
        predictions = torch.ones([batch_size, beam_size, 1],
                                 dtype=torch.int64) * start_id

        if self.use_gpu:
            pos_index = pos_index.cuda()
            predictions = predictions.cuda()

        # initial input (start_id)
        state['pred_token'] = predictions[:, :1]
        if prev_input is not None:
            state['pred_mask'] = torch.not_equal(state['pred_token'],
                                                 self.pad_id).float()
            state['pred_pos'] = state['pred_pos'] + 1

        # shape: [batch_size, vocab_size]
        scores, state = step_fn(state)

        unk_penalty = np.zeros(self.vocab_size, dtype='float32')
        unk_penalty[self.unk_id] = -1e10
        unk_penalty = torch.from_numpy(unk_penalty)

        eos_penalty = np.zeros(self.vocab_size, dtype='float32')
        eos_penalty[eos_id] = -1e10
        eos_penalty = torch.from_numpy(eos_penalty)

        scores_after_end = np.full(self.vocab_size, -1e10, dtype='float32')
        scores_after_end[
            self.
            pad_id] = 0  # we want <pad> is generated after <eos>，so maximum log(p(<pad>)) is (0)
        scores_after_end = torch.from_numpy(scores_after_end)

        if self.use_gpu:
            unk_penalty = unk_penalty.cuda()
            eos_penalty = eos_penalty.cuda()
            scores_after_end = scores_after_end.cuda()
        if self.ignore_unk:
            scores = scores + unk_penalty
        scores = scores + eos_penalty

        # shape: [batch_size, beam_size]
        sequence_scores, preds = torch.topk(scores, self.beam_size)

        predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)
        state = repeat(state, beam_size)

        if max_gen_len is None:
            max_gen_len = self.max_gen_len
        for step in range(2, max_gen_len + 1):
            pre_ids = predictions[:, :, -1:]
            state['pred_token'] = pre_ids.reshape(batch_size * beam_size, 1, 1)
            state['pred_mask'] = torch.not_equal(state['pred_token'],
                                                 self.pad_id).float()
            state['pred_pos'] = state['pred_pos'] + 1
            scores, state = step_fn(state)

            # Generate next
            # scores shape: [batch_size * beam_size, vocab_size]
            if self.ignore_unk:
                scores = scores + unk_penalty

            if step <= self.min_gen_len:
                scores = scores + eos_penalty

            # scores shape: [batch_size, beam_size, vocab_size]
            scores = scores.reshape(batch_size, beam_size, self.vocab_size)

            # previous token is [PAD] or [EOS]
            pre_eos_mask = (1 - torch.not_equal(pre_ids, eos_id).float()) + \
                           (1 - torch.not_equal(pre_ids, self.pad_id).float())

            scores = scores * (1 - pre_eos_mask) + pre_eos_mask.repeat(
                1, 1, self.vocab_size) * scores_after_end
            if self.length_average:
                scaled_value = \
                    pre_eos_mask + (1 - pre_eos_mask) * (1 - 1 / step)
                sequence_scores = sequence_scores.unsqueeze(2) * scaled_value
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 / step)
                scores = scores * scaled_value
            elif self.length_penalty >= 0.0:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow((4 + step) / (5 + step), self.length_penalty))
                sequence_scores = scaled_value * sequence_scores
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow(1 / (5 + step), self.length_penalty))
                scores = scores * scaled_value
            scores = scores + sequence_scores.unsqueeze(-1)
            scores = scores.reshape(batch_size, beam_size * self.vocab_size)

            topk_scores, topk_indices = torch.topk(scores, beam_size)
            # topk_indices: [batch_size, beam_size * self.vocab_size] (already reshaped)
            parent_idx = topk_indices.floor_divide(self.vocab_size)
            preds = topk_indices % self.vocab_size

            # Gather state / sequence_scores
            parent_idx = parent_idx + pos_index
            parent_idx = parent_idx.reshape(batch_size * beam_size)
            state = gather(state, parent_idx)
            sequence_scores = topk_scores

            predictions = predictions.reshape(batch_size * beam_size, step)
            predictions = gather(predictions, parent_idx)
            predictions = predictions.reshape(batch_size, beam_size, step)
            predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)

        # The last token should be <eos> or <pad>
        pre_ids = predictions[:, :, -1]
        pre_eos_mask = (1 - torch.not_equal(pre_ids, eos_id).float()) + \
                       (1 - torch.not_equal(pre_ids, self.pad_id).float())
        sequence_scores = sequence_scores * pre_eos_mask + (
            1 - pre_eos_mask) * (-1e10)

        # first get ascending ordered index，then sort "predictions" and "sequence_scores"
        indices = torch.argsort(sequence_scores, dim=1)
        indices = indices + pos_index
        indices = indices.reshape(-1)
        sequence_scores = sequence_scores.reshape(batch_size * beam_size)
        predictions = predictions.reshape(batch_size * beam_size, -1)
        sequence_scores = gather(sequence_scores, indices)
        predictions = gather(predictions, indices)
        sequence_scores = sequence_scores.reshape(batch_size, beam_size)
        predictions = predictions.reshape(batch_size, beam_size, -1)

        results = {
            'preds': predictions[:, -1],
            'scores': sequence_scores[:, -1]
        }
        return results


BeamSearch.register('BeamSearch')

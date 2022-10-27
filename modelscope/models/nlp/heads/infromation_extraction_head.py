# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from torch import nn

from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.utils.constant import Tasks


@HEADS.register_module(
    Tasks.information_extraction, module_name=Heads.information_extraction)
@HEADS.register_module(
    Tasks.relation_extraction, module_name=Heads.information_extraction)
class InformationExtractionHead(TorchHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        assert config.get('labels') is not None
        self.labels = config.labels
        self.s_layer = nn.Linear(config.hidden_size, 2)  # head, tail, bce
        self.o_layer = nn.Linear(2 * config.hidden_size, 2)  # head, tail, bce
        self.p_layer = nn.Linear(config.hidden_size,
                                 len(self.labels))  # label, ce
        self.mha = nn.MultiheadAttention(config.hidden_size, 4)

    def forward(self, sequence_output, text, offsets, threshold=0.5):
        # assert batch size == 1
        spos = []
        s_head_logits, s_tail_logits = self.s_layer(sequence_output).split(
            1, dim=-1)  # (b, seq_len, 2)
        s_head_logits = s_head_logits[0, :, 0].sigmoid()  # (seq_len)
        s_tail_logits = s_tail_logits[0, :, 0].sigmoid()  # (seq_len)
        s_masks, subjects = self._get_masks_and_mentions(
            text, offsets, s_head_logits, s_tail_logits, None, threshold)
        for s_mask, subject in zip(s_masks, subjects):
            masked_sequence_output = sequence_output * s_mask.unsqueeze(
                0).unsqueeze(-1)  # (b, s, h)
            subjected_sequence_output = self.mha(
                sequence_output.permute(1, 0, 2),
                masked_sequence_output.permute(1, 0, 2),
                masked_sequence_output.permute(1, 0,
                                               2))[0].permute(1, 0,
                                                              2)  # (b, s, h)
            cat_sequence_output = torch.cat(
                (sequence_output, subjected_sequence_output), dim=-1)
            o_head_logits, o_tail_logits = self.o_layer(
                cat_sequence_output).split(
                    1, dim=-1)
            o_head_logits = o_head_logits[0, :, 0].sigmoid()  # (seq_len)
            o_tail_logits = o_tail_logits[0, :, 0].sigmoid()  # (seq_len)
            so_masks, objects = self._get_masks_and_mentions(
                text, offsets, o_head_logits, o_tail_logits, s_mask, threshold)
            for so_mask, object in zip(so_masks, objects):
                masked_sequence_output = (
                    sequence_output * so_mask.unsqueeze(0).unsqueeze(-1)).sum(
                        1)  # (b, h)
                lengths = so_mask.unsqueeze(0).sum(-1, keepdim=True)  # (b, 1)
                pooled_subject_object = masked_sequence_output / lengths  # (b, h)
                label = self.p_layer(pooled_subject_object).sigmoid().squeeze(
                    0)
                for i in range(label.size(-1)):
                    if label[i] > threshold:
                        predicate = self.labels[i]
                        spos.append((subject, predicate, object))
        return spos

    def _get_masks_and_mentions(self,
                                text,
                                offsets,
                                heads,
                                tails,
                                init_mask=None,
                                threshold=0.5):
        '''
        text: str
        heads: tensor (len(heads))
        tails: tensor (len(tails))
        '''
        seq_len = heads.size(-1)
        potential_heads = []
        for i in range(seq_len - 1):
            if heads[i] > threshold:
                potential_heads.append(i)
        potential_heads.append(seq_len - 1)
        masks = []
        mentions = []
        for i in range(len(potential_heads) - 1):
            head_index = potential_heads[i]
            tail_index, max_val = None, 0
            for j in range(head_index, potential_heads[i + 1]):
                if tails[j] > max_val and tails[j] > threshold:
                    tail_index = j
                    max_val = tails[j]
            if tail_index is not None:
                mask = torch.zeros_like(
                    heads) if init_mask is None else init_mask.clone()
                mask[head_index:tail_index + 1] = 1
                masks.append(mask)  # (seq_len)
                char_head = offsets[head_index][0]
                char_tail = offsets[tail_index][1]
                mention = text[char_head:char_tail]
                mentions.append(mention)
        return masks, mentions

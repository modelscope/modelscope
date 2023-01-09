# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List

import numpy as np
import torch


def collate_fn(samples, pad_idx, eos_idx):
    r"""
    convert the sample to batch tensor.
    """
    if len(samples) == 0:
        return {}

    def merge(key):
        return collate_tokens([s[key] for s in samples],
                              pad_idx,
                              eos_idx=eos_idx)

    batch = {
        'nsentences': len(samples),
        'net_input': {},
    }
    if samples[0].get('source', None) is not None:
        batch['net_input']['input_ids'] = merge('source')
    if samples[0].get('id', None) is not None:
        batch['id'] = np.array([s.get('id') for s in samples])
    if samples[0].get('target', None) is not None:
        batch['target'] = merge('target')
        tgt_lengths = torch.LongTensor(
            [s['target'].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()
        batch['ntokens'] = ntokens
    if samples[0].get('prev_output_tokens', None) is not None:
        batch['net_input']['decoder_input_ids'] = merge('prev_output_tokens')
    if samples[0].get('patch_image', None) is not None:
        batch['net_input']['patch_images'] = torch.stack(
            [sample['patch_image'] for sample in samples], dim=0)
    if samples[0].get('patch_mask', None) is not None:
        batch['net_input']['patch_masks'] = torch.cat(
            [sample['patch_mask'] for sample in samples])
    # image generation
    if samples[0].get('code_mask', None) is not None:
        batch['net_input']['code_masks'] = torch.cat(
            [sample['code_mask'] for sample in samples])
    if samples[0].get('code_image', None) is not None:
        batch['code_images'] = torch.cat(
            [sample['code_image'] for sample in samples])
    # For classification tasks (i.e., VQA, SNLI-VE, GLUE)
    if samples[0].get('conf', None) is not None:
        batch['conf'] = torch.cat([s['conf'] for s in samples], dim=0)
    if samples[0].get('ref_dict', None) is not None:
        batch['ref_dict'] = np.array([s['ref_dict'] for s in samples])
    if samples[0].get('label', None) is not None:
        batch['labels'] = np.array([s['label'] for s in samples]).tolist()
    if samples[0].get('constraint_mask', None) is not None:
        batch['constraint_masks'] = merge('constraint_mask')
    if samples[0].get('decoder_prompt', None) is not None:
        batch['decoder_prompts'] = np.array(
            [s['decoder_prompt'].tolist() for s in samples])
    if samples[0].get('prefix_token', None) is not None:
        batch['prefix_tokens'] = merge('prefix_token')
    # For detection and visual grounding
    if samples[0].get('w_resize_ratio', None) is not None:
        batch['w_resize_ratios'] = torch.stack(
            [s['w_resize_ratio'] for s in samples], dim=0)
    if samples[0].get('h_resize_ratio', None) is not None:
        batch['h_resize_ratios'] = torch.stack(
            [s['h_resize_ratio'] for s in samples], dim=0)
    if samples[0].get('region_coord', None) is not None:
        batch['region_coords'] = torch.stack(
            [s['region_coord'] for s in samples], dim=0)
    if samples[0].get('sample', None) is not None:
        batch['samples'] = [s['sample'] for s in samples]
    # For asr
    if samples[0].get('fbank', None) is not None:
        batch['net_input']['fbank'] = _collate_frames(
            [s['fbank'] for s in samples])
        batch['net_input']['fbank_length'] = torch.tensor(
            [s['fbank'].size(0) for s in samples], dtype=torch.long)
    if samples[0].get('fbank_mask', None) is not None:
        batch['net_input']['fbank_masks'] = torch.cat(
            [s['fbank_mask'] for s in samples])
    if samples[0].get('phone_item', None) is not None:
        batch['net_input']['phone_items'] = merge('phone_item')
        batch['net_input']['phone_masks'] = torch.cat(
            [s['phone_mask'] for s in samples])
    if samples[0].get('phone_target', None) is not None:
        batch['phone_target'] = merge('phone_target')
        batch['phone_length'] = torch.tensor(
            [s['phone_target'].size(0) for s in samples], dtype=torch.long)

    # for sudoku
    if samples[0].get('db_struct', None) is not None:
        db_struct = [sample['db_struct'] for sample in samples]
        batch['db_struct'] = db_struct
    if samples[0].get('mask_ratio', None) is not None:
        mask_ratio = [sample['mask_ratio'] for sample in samples]
        batch['mask_ratio'] = mask_ratio
    if samples[0].get('seg_col_tokens', None) is not None:
        seg_col_tokens = merge('seg_col_tokens')
        batch['net_input']['seg_col_tokens'] = seg_col_tokens
    if samples[0].get('seg_row_tokens', None) is not None:
        seg_row_tokens = merge('seg_row_tokens')
        batch['net_input']['seg_row_tokens'] = seg_row_tokens

    return batch


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), size,
                            values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def _collate_frames(frames: List[torch.Tensor]):
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, :v.size(0)] = v
    return out

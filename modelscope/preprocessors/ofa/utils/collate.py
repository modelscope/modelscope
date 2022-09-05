import numpy as np
import torch


def collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return collate_tokens([s[key] for s in samples],
                              pad_idx,
                              eos_idx=eos_idx)

    src_tokens = merge('source')

    batch = {
        'nsentences': len(samples),
        'net_input': {
            'input_ids': src_tokens,
        },
    }
    if samples[0].get('id', None) is not None:
        batch['id'] = np.array([s.get['id'] for s in samples])
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

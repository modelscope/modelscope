# The implementation is adopted from OSTrack,
# made publicly available under the MIT License at https://github.com/botaoye/OSTrack/
import torch


def combine_tokens(template_tokens,
                   search_tokens,
                   mode='direct',
                   return_res=False):
    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    else:
        raise NotImplementedError

    return merged_feature


def recover_tokens(merged_tokens, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError

    return recovered_tokens

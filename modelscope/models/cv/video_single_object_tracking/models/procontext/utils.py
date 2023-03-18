# The ProContEXT implementation is also open-sourced by the authors,
# and available at https://github.com/jp-lan/ProContEXT
import torch


def combine_multi_tokens(template_tokens, search_tokens, mode='direct'):
    if mode == 'direct':
        if not isinstance(template_tokens, list):
            merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
        elif len(template_tokens) >= 2:
            merged_feature = torch.cat(
                (template_tokens[0], template_tokens[1]), dim=1)
            for i in range(2, len(template_tokens)):
                merged_feature = torch.cat(
                    (merged_feature, template_tokens[i]), dim=1)
            merged_feature = torch.cat((merged_feature, search_tokens), dim=1)
        else:
            merged_feature = torch.cat(
                (template_tokens[0], template_tokens[1]), dim=1)
    else:
        raise NotImplementedError
    return merged_feature

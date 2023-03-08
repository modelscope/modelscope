# The ProContEXT implementation is also open-sourced by the authors,
# and available at https://github.com/jp-lan/ProContEXT
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from modelscope.models.cv.video_single_object_tracking.models.layers.attn_blocks import \
    CEBlock
from modelscope.models.cv.video_single_object_tracking.models.layers.patch_embed import \
    PatchEmbed
from modelscope.models.cv.video_single_object_tracking.models.ostrack.utils import (
    combine_tokens, recover_tokens)
from modelscope.models.cv.video_single_object_tracking.models.ostrack.vit_ce import \
    VisionTransformerCE
from .utils import combine_multi_tokens


class VisionTransformerCE_ProContEXT(VisionTransformerCE):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def forward_features(
        self,
        z,
        x,
        mask_x=None,
        ce_template_mask=None,
        ce_keep_rate=None,
    ):
        B = x.shape[0]

        x = self.patch_embed(x)
        x += self.pos_embed_x
        if not isinstance(z, list):
            z = self.patch_embed(z)
            z += self.pos_embed_z
            lens_z = self.pos_embed_z.shape[1]
            x = combine_tokens(z, x, mode=self.cat_mode)
        else:
            z_list = []
            for zi in z:
                z_list.append(self.patch_embed(zi) + self.pos_embed_z)
            lens_z = self.pos_embed_z.shape[1] * len(z_list)
            x = combine_multi_tokens(z_list, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]],
                                device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=x)

        x = recover_tokens(x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            'attn': attn,
            'removed_indexes_s': removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None):

        x, aux_dict = self.forward_features(
            z,
            x,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
        )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE_ProContEXT(**kwargs)
    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

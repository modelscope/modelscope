# The implementation is adopted from OSTrack,
# made publicly available under the MIT License at https://github.com/botaoye/OSTrack/
import torch.nn as nn
from timm.models.layers import to_2tuple

from modelscope.models.cv.video_single_object_tracking.models.layers.patch_embed import \
    PatchEmbed


class BaseBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_stage = [2, 5, 8, 11]

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print(
                'Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!'
            )
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(
                        param,
                        size=(new_patch_size, new_patch_size),
                        mode='bicubic',
                        align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(
                img_size=self.img_size,
                patch_size=new_patch_size,
                in_chans=3,
                embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[
            1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(
            1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(
            2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

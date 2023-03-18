# The ProContEXT implementation is also open-sourced by the authors,
# and available at https://github.com/jp-lan/ProContEXT
import torch
from torch import nn

from modelscope.models.cv.video_single_object_tracking.models.layers.head import \
    build_box_head
from .vit_ce import vit_base_patch16_224_ce


class ProContEXT(nn.Module):
    """ This is the base class for ProContEXT """

    def __init__(self,
                 transformer,
                 box_head,
                 aux_loss=False,
                 head_type='CORNER'):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == 'CORNER' or head_type == 'CENTER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz**2)

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        ce_template_mask=None,
        ce_keep_rate=None,
    ):
        x, aux_dict = self.backbone(
            z=template,
            x=search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
        )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.
                              feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == 'CENTER':
            # run the center head
            score_map_ctr, bbox, size_map, offset_map, score = self.box_head(
                opt_feat, return_score=True)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map,
                'score': score
            }
            return out
        else:
            raise NotImplementedError


def build_procontext(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(
            False,
            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ProContEXT(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    return model

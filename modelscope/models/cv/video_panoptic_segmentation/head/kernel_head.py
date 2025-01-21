# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .semantic_fpn_wrapper import SemanticFPNWrapper


class ConvKernelHead(nn.Module):

    def __init__(self,
                 num_proposals=100,
                 in_channels=256,
                 out_channels=256,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_seg_convs=1,
                 num_loc_convs=1,
                 att_dropout=False,
                 conv_kernel_size=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 semantic_fpn=True,
                 train_cfg=None,
                 num_classes=80,
                 xavier_init_kernel=False,
                 kernel_init_std=0.01,
                 use_binary=False,
                 proposal_feats_with_obj=False,
                 feat_downsample_stride=1,
                 feat_refine_stride=1,
                 feat_refine=True,
                 with_embed=False,
                 feat_embed_only=False,
                 conv_normal_init=False,
                 mask_out_stride=4,
                 hard_target=False,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cat_stuff_mask=False,
                 **kwargs):
        super(ConvKernelHead, self).__init__()
        self.num_proposals = num_proposals
        self.num_cls_fcs = num_cls_fcs
        self.train_cfg = train_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.proposal_feats_with_obj = proposal_feats_with_obj
        self.sampling = False
        self.localization_fpn = SemanticFPNWrapper(
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            upsample_times=2,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            cat_coors=False,
            cat_coors_level=3,
            fuse_by_cat=False,
            return_list=False,
            num_aux_convs=1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
        self.semantic_fpn = semantic_fpn
        self.norm_cfg = norm_cfg
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.conv_kernel_size = conv_kernel_size
        self.xavier_init_kernel = xavier_init_kernel
        self.kernel_init_std = kernel_init_std
        self.feat_downsample_stride = feat_downsample_stride
        self.feat_refine_stride = feat_refine_stride
        self.conv_normal_init = conv_normal_init
        self.feat_refine = feat_refine
        self.with_embed = with_embed
        self.feat_embed_only = feat_embed_only
        self.num_loc_convs = num_loc_convs
        self.num_seg_convs = num_seg_convs
        self.use_binary = use_binary
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg
        self.cat_stuff_mask = cat_stuff_mask
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_kernels = nn.Conv2d(
            self.out_channels,
            self.num_proposals,
            self.conv_kernel_size,
            padding=int(self.conv_kernel_size // 2),
            bias=False)  # (N, C)

        if self.semantic_fpn:
            self.conv_seg = nn.Conv2d(self.out_channels, self.num_classes, 1)

        if self.feat_downsample_stride > 1 and self.feat_refine:
            self.ins_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,  # 2
                padding=1,
                norm_cfg=self.norm_cfg)
            self.seg_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,  # 2
                padding=1,
                norm_cfg=self.norm_cfg)

        self.loc_convs = nn.ModuleList()
        for i in range(self.num_loc_convs):
            self.loc_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.seg_convs = nn.ModuleList()
        for i in range(self.num_seg_convs):
            self.seg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

    def _decode_init_proposals(self, img, img_metas):
        num_imgs = len(img_metas)

        localization_feats = self.localization_fpn(img)

        # thing branch
        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        for conv in self.loc_convs:
            loc_feats = conv(loc_feats)
        if self.feat_downsample_stride > 1 and self.feat_refine:
            loc_feats = self.ins_downsample(loc_feats)

        # init kernel prediction
        mask_preds = self.init_kernels(loc_feats)

        # stuff branch
        if self.semantic_fpn:
            if isinstance(localization_feats, list):
                semantic_feats = localization_feats[1]
            else:
                semantic_feats = localization_feats
            for conv in self.seg_convs:
                semantic_feats = conv(semantic_feats)
            if self.feat_downsample_stride > 1 and self.feat_refine:
                semantic_feats = self.seg_downsample(semantic_feats)
        else:
            semantic_feats = None

        if semantic_feats is not None:
            seg_preds = self.conv_seg(semantic_feats)
        else:
            seg_preds = None

        proposal_feats = self.init_kernels.weight.clone()
        proposal_feats = proposal_feats[None].expand(num_imgs,
                                                     *proposal_feats.size())

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats
        else:
            x_feats = loc_feats

        if self.proposal_feats_with_obj:
            sigmoid_masks = mask_preds.sigmoid()
            nonzero_inds = sigmoid_masks > 0.5
            if self.use_binary:
                sigmoid_masks = nonzero_inds.float()
            else:
                sigmoid_masks = nonzero_inds.float() * sigmoid_masks
            obj_feats = torch.einsum('bnhw, bchw->bnc', sigmoid_masks, x_feats)

        cls_scores = None

        if self.proposal_feats_with_obj:  # important use
            proposal_feats = proposal_feats + obj_feats.view(
                num_imgs, self.num_proposals, self.out_channels, 1, 1)

        if self.cat_stuff_mask and not self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels],
                                       dim=1)  # (b, N_{st}+N_{th}, c)

        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)

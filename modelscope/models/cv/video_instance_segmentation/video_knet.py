# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net follow the MIT license

import torch.nn as nn
from mmdet.models import build_head, build_neck

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_panoptic_segmentation.backbone.swin_transformer import \
    SwinTransformerDIY
from modelscope.models.cv.video_panoptic_segmentation.head.semantic_fpn_wrapper import \
    SemanticFPNWrapper
from modelscope.utils.constant import Tasks
from .head.kernel_frame_iter_head import KernelFrameIterHeadVideo
from .head.kernel_head import ConvKernelHeadVideo
from .head.kernel_iter_head import KernelIterHeadVideo
from .head.kernel_update_head import KernelUpdateHead
from .head.kernel_updator import KernelUpdator
from .neck import MSDeformAttnPixelDecoder
from .track.kernel_update_head import KernelUpdateHeadVideo
from .track.mask_hungarian_assigner import MaskHungarianAssignerVideo


@MODELS.register_module(
    Tasks.video_instance_segmentation,
    module_name=Models.video_instance_segmentation)
class KNetTrack(TorchModel):
    """
    Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation (https://arxiv.org/pdf/2204.04656.pdf)
    Video K-Net is a strong and unified framework for fully end-to-end video panoptic and instance segmentation.
    The method is built upon K-Net, a method that unifies image segmentation via a group of learnable kernels.
    K-Net learns to simultaneously segment and track “things” and “stuff” in a video with simple kernel-based
    appearance modeling and cross-temporal kernel interaction.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)

        self.roi_head = None
        num_stages = 3
        num_proposals = 100
        conv_kernel_size = 1
        num_thing_classes = 40
        num_stuff_classes = 0
        mask_assign_stride = 4
        thing_label_in_seg = 0
        direct_tracker = False
        tracker_num = 1

        # assert self.with_rpn, 'KNet does not support external proposals'
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.direct_tracker = direct_tracker
        self.tracker_num = tracker_num

        train_cfg = dict(
            rpn=dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaskHungarianAssigner',
                        cls_cost=dict(type='FocalLossCost', weight=2.0),
                        dice_cost=dict(
                            type='DiceCost', weight=4.0, pred_act=True),
                        mask_cost=dict(
                            type='MaskCost', weight=1.0, pred_act=True)),
                    sampler=dict(type='MaskPseudoSampler'),
                    pos_weight=1) for _ in range(num_stages)
            ],
            tracker=dict(
                assigner=dict(
                    type='MaskHungarianAssignerVideo',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1))
        self.train_cfg = train_cfg

        test_cfg = dict(
            rpn=None,
            rcnn=dict(
                max_per_img=10,
                mask_thr=0.5,
                merge_stuff_thing=dict(
                    iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3)),
            tracker=dict(
                max_per_img=10,
                mask_thr=0.5,
                merge_stuff_thing=dict(
                    iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3),
            ))
        self.test_cfg = test_cfg

        self.backbone = SwinTransformerDIY(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=True)

        neck = dict(
            type='MSDeformAttnPixelDecoder',
            in_channels=[128, 256, 512, 1024],
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            return_one_list=True,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None)
        self.neck = build_neck(neck)

        rpn_head = dict(
            type='ConvKernelHeadVideo',
            conv_kernel_size=conv_kernel_size,
            feat_downsample_stride=2,
            feat_refine_stride=1,
            feat_refine=False,
            use_binary=True,
            num_loc_convs=1,
            num_seg_convs=1,
            conv_normal_init=True,
            localization_fpn=dict(
                type='SemanticFPNWrapper',
                in_channels=256,
                feat_channels=256,
                out_channels=256,
                start_level=0,
                end_level=3,
                upsample_times=2,
                positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128,
                    normalize=True),
                cat_coors=False,
                cat_coors_level=3,
                fuse_by_cat=False,
                return_list=False,
                num_aux_convs=1,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
            num_proposals=num_proposals,
            proposal_feats_with_obj=True,
            xavier_init_kernel=False,
            kernel_init_std=1,
            num_cls_fcs=1,
            in_channels=256,
            num_classes=40,
            feat_transform_cfg=None,
            loss_seg=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_mask=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_dice=dict(type='DiceLoss', loss_weight=4.0))

        self.rpn_head = build_head(rpn_head)

        roi_head = dict(
            type='KernelIterHeadVideo',
            num_stages=num_stages,
            stage_loss_weights=[1] * num_stages,
            proposal_feature_channel=256,
            num_thing_classes=40,
            num_stuff_classes=0,
            mask_head=[
                dict(
                    type='KernelUpdateHead',
                    num_classes=40,
                    num_thing_classes=40,
                    num_stuff_classes=0,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_cls_fcs=1,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=256,
                    out_channels=256,
                    dropout=0.0,
                    mask_thr=0.5,
                    conv_kernel_size=conv_kernel_size,
                    mask_upsample_stride=2,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        input_feat_shape=3,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')),
                    loss_mask=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0)) for _ in range(num_stages)
            ])
        roi_head.update(test_cfg=self.test_cfg['rcnn'])
        self.roi_head = build_head(roi_head)

        tracker = dict(
            type='KernelFrameIterHeadVideo',
            num_proposals=num_proposals,
            num_stages=3,
            assign_stages=2,
            proposal_feature_channel=256,
            stage_loss_weights=(1., 1., 1.),
            num_thing_classes=40,
            num_stuff_classes=0,
            mask_head=dict(
                type='KernelUpdateHeadVideo',
                num_proposals=num_proposals,
                num_classes=40,
                num_thing_classes=40,
                num_stuff_classes=0,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)))

        if tracker is not None:
            rcnn_train_cfg = train_cfg[
                'tracker'] if train_cfg is not None else None
            tracker.update(train_cfg=rcnn_train_cfg)
            tracker.update(test_cfg=test_cfg['tracker'])
            self.tracker = build_head(tracker)
            if self.tracker_num > 1:
                self.tracker_extra = nn.ModuleList(
                    [build_head(tracker) for _ in range(tracker_num - 1)])

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            kwargs['ref_img_metas'] = kwargs['ref_img_metas'][0]
            kwargs['ref_img'] = kwargs['ref_img'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test(self, imgs, img_metas, **kwargs):
        ref_img = kwargs['ref_img']
        ref_img_metas = kwargs['ref_img_metas']
        # Step 1 extract features and get masks
        bs, num_frame, _, h, w = ref_img.size()
        x = self.extract_feat(ref_img.reshape(bs * num_frame, _, h, w))

        proposal_feats, x_feats, mask_preds, cls_scores, seg_preds = \
            self.rpn_head.simple_test_rpn(x, img_metas, ref_img_metas)

        if self.roi_head is not None:
            segm_results_single_frame, features = self.roi_head.simple_test(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas,
                ref_img_metas,
                imgs_whwh=None,
                rescale=True)

        if self.direct_tracker:
            proposal_feats = self.rpn_head.init_kernels.weight.clone()
            proposal_feats = proposal_feats[None].expand(
                bs, *proposal_feats.size())
            if mask_preds.shape[0] == bs * num_frame:
                mask_preds = mask_preds.reshape(
                    (bs, num_frame, *mask_preds.size()[1:]))
                x_feats = x_feats.reshape((bs, num_frame, *x_feats.size()[1:]))
            else:
                assert mask_preds.size()[:2] == (bs, num_frame)
                assert x_feats.size()[:2] == (bs, num_frame)
            segm_results, features = self.tracker.simple_test(
                x=x_feats,
                img_metas=img_metas,
                ref_img_metas=ref_img_metas,
                cls_scores=None,
                masks=mask_preds,
                obj_feats=proposal_feats,
            )
            if self.tracker_num > 1:
                for i in range(self.tracker_num - 1):
                    segm_results, features = self.tracker_extra[i].simple_test(
                        x=features['x_feats'],
                        img_metas=img_metas,
                        ref_img_metas=ref_img_metas,
                        cls_scores=None,
                        masks=features['masks'],
                        obj_feats=features['obj_feats'],
                    )
        else:
            segm_results, _ = self.tracker.simple_test(
                x=features['x_feats'],
                img_metas=img_metas,
                ref_img_metas=ref_img_metas,
                cls_scores=features['cls_scores'],
                masks=features['masks'],
                obj_feats=features['obj_feats'],
            )

        return segm_results

# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet.models.builder import build_head

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .backbone.swin_transformer import SwinTransformerDIY
from .head.kernel_head import ConvKernelHead
from .head.kernel_iter_head import VideoKernelIterHead
from .head.track_heads import QuasiDenseMaskEmbedHeadGTMask
from .neck.fpn import FPN
from .track.quasi_dense_embed_tracker import (QuasiDenseEmbedTracker,
                                              build_tracker)
from .visualizer import vip_seg_id_to_label


def coords2bbox_all(coords):
    left = coords[:, 0].min().item()
    top = coords[:, 1].min().item()
    right = coords[:, 0].max().item()
    bottom = coords[:, 1].max().item()
    return top, left, bottom, right


def tensor_mask2box(masks):
    boxes = []
    for mask in masks:
        m = mask.nonzero().float()
        if m.numel() > 0:
            box = coords2bbox_all(m)
        else:
            box = (-1, -1, 10, 10)
        boxes.append(box)
    return np.asarray(boxes)


@MODELS.register_module(
    Tasks.video_panoptic_segmentation,
    module_name=Models.video_panoptic_segmentation)
class VideoKNet(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)

        num_proposals = 100
        num_stages = 3
        conv_kernel_size = 1
        num_thing_classes = 58
        num_stuff_classes = 66
        num_classes = num_thing_classes + num_stuff_classes

        self.num_proposals = num_proposals
        self.num_stages = num_stages
        self.conv_kernel_size = conv_kernel_size
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_classes

        self.semantic_filter = True
        self.link_previous = True
        self.kitti_step = False
        self.cityscapes = False
        self.vipseg = True

        self.test_cfg = dict(
            rpn=None,
            rcnn=dict(
                max_per_img=num_proposals,
                mask_thr=0.5,
                stuff_score_thr=0.05,
                merge_stuff_thing=dict(
                    overlap_thr=0.6,
                    iou_thr=0.5,
                    stuff_max_area=4096,
                    instance_score_thr=0.25)))

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
            with_cp=False)

        self.neck = FPN(
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=4)

        self.rpn_head = ConvKernelHead(
            conv_kernel_size=conv_kernel_size,
            feat_downsample_stride=4,
            feat_refine_stride=1,
            feat_refine=False,
            use_binary=True,
            num_loc_convs=1,
            num_seg_convs=1,
            conv_normal_init=True,
            num_proposals=num_proposals,
            proposal_feats_with_obj=True,
            xavier_init_kernel=False,
            kernel_init_std=1,
            num_cls_fcs=1,
            in_channels=256,
            num_thing_classes=num_thing_classes,
            num_stuff_classes=num_stuff_classes,
            num_classes=num_classes,
            cat_stuff_mask=True,
            feat_transform_cfg=None)

        roi_head = dict(
            type='VideoKernelIterHead',
            num_stages=num_stages,
            stage_loss_weights=[1] * num_stages,
            proposal_feature_channel=256,
            num_thing_classes=num_thing_classes,
            num_stuff_classes=num_stuff_classes,
            do_panoptic=True,
            with_track=True,
            merge_joint=True,
            mask_head=[
                dict(
                    type='VideoKernelUpdateHead',
                    num_classes=num_classes,
                    previous='placeholder',
                    previous_type='ffn',
                    num_thing_classes=num_thing_classes,
                    num_stuff_classes=num_stuff_classes,
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
                    mask_upsample_stride=4,
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
                        loss_weight=2.0),
                ) for _ in range(num_stages)
            ])
        roi_head.update(test_cfg=self.test_cfg['rcnn'])
        self.roi_head = build_head(roi_head)

        self.track_head = QuasiDenseMaskEmbedHeadGTMask(
            num_convs=0,
            num_fcs=2,
            roi_feat_size=1,
            in_channels=256,
            fc_out_channels=256,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32))

        self.tracker_cfg = dict(
            type='QuasiDenseEmbedTracker',
            init_score_thr=0.35,
            obj_score_thr=0.3,
            match_score_thr=0.5,
            memo_tracklet_frames=5,
            memo_backdrop_frames=1,
            memo_momentum=0.8,
            nms_conf_thr=0.5,
            nms_backdrop_iou_thr=0.3,
            nms_class_iou_thr=0.7,
            with_cats=True,
            match_metric='bisoftmax')

        # add embedding fcs for the final stage queries
        num_emb_fcs = 1
        act_cfg = dict(type='ReLU', inplace=True)
        in_channels = 256
        out_channels = 256
        self.embed_fcs = nn.ModuleList()
        for _ in range(num_emb_fcs):
            self.embed_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.embed_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.embed_fcs.append(build_activation_layer(act_cfg))

        self.fc_embed = nn.Linear(in_channels, out_channels)

        self.link_previous = True,

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def forward(self, img, img_metas, rescale=False, ref_img=None, iid=0):
        result = self.simple_test(img, img_metas, rescale, ref_img, iid)
        return result

    def simple_test(self, img, img_metas, rescale=False, ref_img=None, iid=0):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        # set the dataset type
        fid = iid % 10000
        is_first = (fid == 0)

        # for current frame
        x = self.extract_feat(img)
        # current frame inference
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results

        # init tracker
        if is_first:
            self.init_tracker()
            self.obj_feats_memory = None
            self.x_feats_memory = None
            self.mask_preds_memory = None
            print('fid', fid)

        # wheter to link the previous
        if self.link_previous:
            simple_test_result = self.roi_head.simple_test_with_previous(
                x_feats,
                proposal_feats,
                mask_preds,
                cls_scores,
                img_metas,
                previous_obj_feats=self.obj_feats_memory,
                previous_mask_preds=self.mask_preds_memory,
                previous_x_feats=self.x_feats_memory,
                is_first=is_first)
            cur_segm_results, obj_feats, cls_scores, mask_preds, scaled_mask_preds = simple_test_result
            self.obj_feats_memory = obj_feats
            self.x_feats_memory = x_feats
            self.mask_preds_memory = scaled_mask_preds
        else:
            cur_segm_results, query_output, cls_scores, mask_preds, scaled_mask_preds = self.roi_head.simple_test(
                x_feats, proposal_feats, mask_preds, cls_scores, img_metas)

        # for tracking part
        _, segm_result, mask_preds, panoptic_result, query_output = cur_segm_results[
            0]
        panoptic_seg, segments_info = panoptic_result

        # get sorted tracking thing ids, labels, masks, score for tracking
        things_index_for_tracking, things_labels_for_tracking, thing_masks_for_tracking, things_score_for_tracking = \
            self.get_things_id_for_tracking(panoptic_seg, segments_info)
        things_labels_for_tracking = torch.Tensor(
            things_labels_for_tracking).to(cls_scores.device).long()

        # get the semantic filter
        if self.semantic_filter:
            seg_preds = torch.nn.functional.interpolate(
                seg_preds,
                panoptic_seg.shape,
                mode='bilinear',
                align_corners=False)
            seg_preds = seg_preds.sigmoid()
            seg_out = seg_preds.argmax(1)
            semantic_thing = (seg_out < self.num_thing_classes).to(
                dtype=torch.float32)
        else:
            semantic_thing = 1.

        if len(things_labels_for_tracking) > 0:
            things_bbox_for_tracking = torch.zeros(
                (len(things_score_for_tracking), 5),
                dtype=torch.float,
                device=x_feats.device)
            things_bbox_for_tracking[:, 4] = torch.tensor(
                things_score_for_tracking,
                device=things_bbox_for_tracking.device)

            thing_masks_for_tracking_final = []
            for mask in thing_masks_for_tracking:
                thing_masks_for_tracking_final.append(
                    torch.Tensor(mask).unsqueeze(0).to(x_feats.device).float())
            thing_masks_for_tracking_final = torch.cat(
                thing_masks_for_tracking_final, 0)
            thing_masks_for_tracking = thing_masks_for_tracking_final
            thing_masks_for_tracking_with_semantic_filter = thing_masks_for_tracking_final * semantic_thing
        else:
            things_bbox_for_tracking = []

        if len(things_labels_for_tracking) == 0:
            track_feats = None
        else:
            # tracking embeddings
            N, _, _, _ = query_output.shape
            emb_feat = query_output.squeeze(-2).squeeze(-1).unsqueeze(
                0)  # (n,d,1,1) -> (1,n,d)

            for emb_layer in self.embed_fcs:
                emb_feat = emb_layer(emb_feat)
            object_feats_embed = self.fc_embed(emb_feat).view(1, N, -1)
            object_feats_embed_for_tracking = object_feats_embed.squeeze(0)
            track_feats = self._track_forward(
                [object_feats_embed_for_tracking])

        if track_feats is not None:
            things_bbox_for_tracking[:, :4] = torch.tensor(
                tensor_mask2box(thing_masks_for_tracking_with_semantic_filter),
                device=things_bbox_for_tracking.device)
            bboxes, labels, ids = self.tracker.match(
                bboxes=things_bbox_for_tracking,
                labels=things_labels_for_tracking,
                track_feats=track_feats,
                frame_id=fid)

            ids = ids + 1
            ids[ids == -1] = 0
        else:
            ids = []

        track_maps = self.generate_track_id_maps(ids, thing_masks_for_tracking,
                                                 panoptic_seg)

        semantic_map, binary_masks, labels = self.get_semantic_seg(
            panoptic_seg, segments_info)

        vis_tracker = None
        vis_sem = None
        from .visualizer import trackmap2rgb, cityscapes_cat2rgb, draw_bbox_on_img
        if len(things_labels_for_tracking):
            vis_tracker = trackmap2rgb(track_maps)
            vis_sem = cityscapes_cat2rgb(semantic_map)
            vis_tracker = draw_bbox_on_img(
                vis_tracker,
                things_bbox_for_tracking.cpu().numpy())

        return semantic_map, track_maps, None, vis_sem, vis_tracker, labels, binary_masks, ids, things_bbox_for_tracking

    def _track_forward(self, track_feats, x=None, mask_pred=None):
        track_feats = torch.cat(track_feats, 0)
        track_feats = self.track_head(track_feats)
        return track_feats

    def get_things_id_for_tracking(self, panoptic_seg, seg_infos):
        idxs = []
        labels = []
        masks = []
        score = []
        for segment in seg_infos:
            if segment['isthing'] is True:
                thing_mask = panoptic_seg == segment['id']
                masks.append(thing_mask)
                idxs.append(segment['instance_id'])
                labels.append(segment['category_id'])
                score.append(segment['score'])
        return idxs, labels, masks, score

    def get_semantic_seg(self, panoptic_seg, segments_info):
        kitti_step2cityscpaes = [11, 13]
        semantic_seg = np.zeros(panoptic_seg.shape)
        binary_masks = []
        labels = []
        for segment in segments_info:
            binary_mask = np.zeros(panoptic_seg.shape)
            if segment['isthing'] is True:
                # for things
                if self.kitti_step:
                    cat_cur = kitti_step2cityscpaes[segment['category_id']]
                    semantic_seg[panoptic_seg == segment['id']] = cat_cur
                    label = cat_cur
                else:  # city and vip_seg
                    semantic_seg[panoptic_seg == segment['id']] = segment[
                        'category_id'] + self.num_stuff_classes
                    label = segment['category_id'] + self.num_stuff_classes
            else:
                # for stuff (0 - n-1)
                if self.kitti_step:
                    cat_cur = segment['category_id']
                    cat_cur -= 1
                    offset = 0
                    for thing_id in kitti_step2cityscpaes:
                        if cat_cur + offset >= thing_id:
                            offset += 1
                    cat_cur += offset
                    semantic_seg[panoptic_seg == segment['id']] = cat_cur
                    label = cat_cur
                else:  # city and vip_seg
                    mask_idx = panoptic_seg == segment['id']
                    semantic_seg[mask_idx] = segment['category_id'] - 1
                    label = segment['category_id'] - 1
            binary_mask[panoptic_seg == segment['id']] = 1
            binary_masks.append(binary_mask)
            labels.append(vip_seg_id_to_label[label])
        return semantic_seg, binary_masks, labels

    def generate_track_id_maps(self, ids, masks, panopitc_seg_maps):
        final_id_maps = np.zeros(panopitc_seg_maps.shape)

        if len(ids) == 0:
            return final_id_maps
        masks = masks.bool()

        for i, id in enumerate(ids):
            mask = masks[i].cpu().numpy()
            final_id_maps[mask] = id

        return final_id_maps

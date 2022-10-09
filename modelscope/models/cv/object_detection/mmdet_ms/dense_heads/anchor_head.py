# Copyright (c) OpenMMLab. All rights reserved.
# Implementation in this file is modified based on ViTAE-Transformer
# Originally Apache 2.0 License and publicly avaialbe at https://github.com/ViTAE-Transformer/ViTDet
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import AnchorHead


@HEADS.register_module()
class AnchorNHead(AnchorHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 anchor_generator=None,
                 bbox_coder=None,
                 reg_decoded_bbox=False,
                 loss_cls=None,
                 loss_bbox=None,
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        self.norm_cfg = norm_cfg
        super(AnchorNHead,
              self).__init__(num_classes, in_channels, feat_channels,
                             anchor_generator, bbox_coder, reg_decoded_bbox,
                             loss_cls, loss_bbox, train_cfg, test_cfg,
                             init_cfg)

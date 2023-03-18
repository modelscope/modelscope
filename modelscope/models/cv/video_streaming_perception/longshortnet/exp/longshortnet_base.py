# Copyright (c) 2014-2021 Megvii Inc.
# Copyright (c) 2022-2023 Alibaba, Inc. and its affiliates. All rights reserved.

from modelscope.models.cv.stream_yolo.exp.yolox_base import Exp


class LongShortNetExp(Exp):

    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 8
        self.test_size = (600, 960)
        self.test_conf = 0.3
        self.nmsthre = 0.65
        self.short_cfg = dict()
        self.long_cfg = dict()
        self.merge_cfg = dict()

    def get_model(self):
        from ..models.longshort import LONGSHORT
        from ..models.dfp_pafpn_long import DFPPAFPNLONG
        from ..models.dfp_pafpn_short import DFPPAFPNSHORT
        from ..models.longshort_backbone_neck import BACKBONENECK
        from modelscope.models.cv.stream_yolo.models.tal_head import TALHead
        import torch.nn as nn

        if getattr(self, 'model', None) is None:
            in_channels = [256, 512, 1024]
            long_backbone = (
                DFPPAFPNLONG(
                    self.depth,
                    self.width,
                    in_channels=in_channels,
                    frame_num=self.long_cfg['frame_num'],
                    with_short_cut=self.long_cfg['with_short_cut'],
                    out_channels=self.long_cfg['out_channels'])
                if self.long_cfg['frame_num'] != 0 else None)
            short_backbone = DFPPAFPNSHORT(
                self.depth,
                self.width,
                in_channels=in_channels,
                frame_num=self.short_cfg['frame_num'],
                with_short_cut=self.short_cfg['with_short_cut'],
                out_channels=self.short_cfg['out_channels'])
            backbone_neck = BACKBONENECK(
                self.depth, self.width, in_channels=in_channels)
            head = TALHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                gamma=1.0,
                ignore_thr=0.5,
                ignore_value=1.5)
            self.model = LONGSHORT(
                long_backbone,
                short_backbone,
                backbone_neck,
                head,
                merge_form=self.merge_cfg['merge_form'],
                in_channels=in_channels,
                width=self.width,
                with_short_cut=self.merge_cfg['with_short_cut'],
                long_cfg=self.long_cfg)
        return self.model

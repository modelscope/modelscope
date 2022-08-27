# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super(YOLOX, self).__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        if self.training:
            raise NotImplementedError('Training is not supported yet!')
        else:
            outputs = self.head(fpn_outs)

        return outputs

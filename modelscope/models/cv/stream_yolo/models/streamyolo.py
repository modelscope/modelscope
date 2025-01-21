# The implementation is based on StreamYOLO, available at https://github.com/yancie-yjr/StreamYOLO
import torch.nn as nn

from .dfp_pafpn import DFPPAFPN
from .tal_head import TALHead


class StreamYOLO(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = DFPPAFPN()
        if head is None:
            head = TALHead(20)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, buffer=None, mode='off_pipe'):
        # fpn output content features of [dark3, dark4, dark5]
        assert mode in ['off_pipe', 'on_pipe']

        if mode == 'off_pipe':
            fpn_outs = self.backbone(x, buffer=buffer, mode='off_pipe')
            if self.training:
                pass
            else:
                outputs = self.head(fpn_outs, imgs=x)

            return outputs
        elif mode == 'on_pipe':
            fpn_outs, buffer_ = self.backbone(x, buffer=buffer, mode='on_pipe')
            outputs = self.head(fpn_outs)

            return outputs, buffer_

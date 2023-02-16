# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn

from modelscope.models.cv.video_deinterlace.models.enh import DeinterlaceEnh
from modelscope.models.cv.video_deinterlace.models.fre import DeinterlaceFre


class DeinterlaceNet(nn.Module):

    def __init__(self):
        super(DeinterlaceNet, self).__init__()
        self.frenet = DeinterlaceFre()
        self.enhnet = DeinterlaceEnh()

    def forward(self, frames):
        self.frenet.eval()
        self.enhnet.eval()
        with torch.no_grad():
            frame1, frame2, frame3 = frames

            F1_out = self.frenet(frame1)
            F2_out = self.frenet(frame2)
            F3_out = self.frenet(frame3)

            out = self.enhnet([F1_out, F2_out, F3_out])

        return out

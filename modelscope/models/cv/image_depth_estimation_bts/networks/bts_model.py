# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class BtsModel(nn.Module):
    """Depth estimation model bts, implemented from paper https://arxiv.org/pdf/1907.10326.pdf.
        The network utilizes novel local planar guidance layers located at multiple stage in the decoding phase.
        The bts model is composed with encoder and decoder, an encoder for dense feature extraction and a decoder
        for predicting the desired depth.
    """

    def __init__(self, focal=715.0873):
        """initial bts model

        Args:
            focal (float): focal length, pictures that do not work are input according to
                the camera setting value at the time of shooting
        """
        super(BtsModel, self).__init__()
        self.focal = focal
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, focal=None):
        """forward to estimation depth

        Args:
            x (Tensor): input image data
            focal (float): The focal length when the picture is taken. By default, the focal length
                of the data set when the model is created is used

        Returns:
            Tensor: Depth estimation image
        """
        focal_run = focal if focal else self.focal
        skip_feat = self.encoder(x)
        depth = self.decoder(skip_feat, torch.tensor(focal_run).cuda())
        return depth

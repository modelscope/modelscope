# The implementation is modified from cleinc / bts
# made publicly available under the GPL-3.0-or-later
# https://github.com/cleinc/bts/blob/master/pytorch/bts.py
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class BtsModel(nn.Module):

    def __init__(self, focal=518.8579):
        """
        initial bts model
        Parameters
        ----------
        focal: focal length, pictures that do not work are input according to
                the camera setting value at the time of shooting
        """
        super(BtsModel, self).__init__()
        self.focal = focal
        self.encoder = Encoder(encoder='densenet161_bts')
        self.decoder = Decoder(
            feat_out_channels=self.encoder.feat_out_channels)

    def forward(self, x, focal=None):
        """
        model forward
        Parameters
        ----------
        x: input image data
        focal: The focal length when the picture is taken. By default, the focal length
                of the data set when the model is created is used

        Returns： Depth estimation image
        -------

        """
        focal_run = focal if focal else self.focal
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal_run)

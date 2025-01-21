# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from .Embedding import Embedding
from .geometry import index, orthogonal, perspective
from .Res_backbone import Res_hournet
from .Surface_head import Surface_Head


class Pixto3DNet(nn.Module):

    def __init__(self,
                 backbone,
                 head,
                 rgbhead,
                 embedding,
                 projection_mode: str = 'orthogonal',
                 error_term: str = 'mse',
                 num_views: int = 1):
        """
        Parameters:
            backbone: parameter of networks to extract image features
            head: parameter of networks to predict value in surface
            rgbhead: parameter of networks to predict rgb of point
            embedding: parameter of networks to normalize depth of camera coordinate
            projection_mode: how to render your 3d model to images
            error_term: train loss
            num_view: how many images from which you want to reconstruct model
        """
        super(Pixto3DNet, self).__init__()

        self.backbone = Res_hournet(**backbone)
        self.head = Surface_Head(**head)
        self.rgbhead = Surface_Head(**rgbhead)
        self.depth = Embedding(**embedding)

        if error_term == 'mse':
            self.error_term = nn.MSELoss(reduction='none')
        elif error_term == 'bce':
            self.error_term = nn.BCELoss(reduction='none')
        elif error_term == 'l1':
            self.error_term = nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.num_views = num_views
        self.im_feat_list = []
        self.intermediate_preds_list = []

    def extract_features(self, images: torch.Tensor):
        self.im_feat_list = self.backbone(images)

    def query(self, points, calibs, transforms=None, labels=None):
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)

        xy = xyz[:, :2, :]
        xyz_feat = self.depth(xyz)

        self.intermediate_preds_list = []

        im_feat_256 = self.im_feat_list[0]
        im_feat_512 = self.im_feat_list[1]

        point_local_feat_list = [
            self.index(im_feat_256, xy),
            self.index(im_feat_512, xy), xyz_feat
        ]
        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred, phi = self.head(point_local_feat)
        self.intermediate_preds_list.append(pred)
        self.phi = phi

        self.preds = self.intermediate_preds_list[-1]

    def get_preds(self):
        return self.preds

    def query_rgb(self, points, calibs, transforms=None):
        xyz = self.projection(points, calibs, transforms)

        xy = xyz[:, :2, :]
        xyz_feat = self.depth(xyz)

        self.intermediate_preds_list = []

        im_feat_256 = self.im_feat_list[0]
        im_feat_512 = self.im_feat_list[1]

        point_local_feat_list = [
            self.index(im_feat_256, xy),
            self.index(im_feat_512, xy), xyz_feat
        ]
        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred, phi = self.head(point_local_feat)
        rgb_point_feat = torch.cat([point_local_feat, phi], 1)
        rgb, phi = self.rgbhead(rgb_point_feat)
        return rgb

    def get_error(self):
        error = 0
        lc = torch.tensor(self.labels.shape[0] * self.labels.shape[1]
                          * self.labels.shape[2])
        inw = torch.sum(self.labels)
        weight_in = inw / lc
        weight = torch.abs(self.labels - weight_in)
        lamda = 1 / torch.mean(weight)
        for preds in self.intermediate_preds_list:
            error += lamda * torch.mean(
                self.error_term(preds, self.labels) * weight)
        error /= len(self.intermediate_preds_list)

        return error

    def forward(self,
                images,
                points,
                calibs,
                surpoint=None,
                transforms=None,
                labels=None):
        self.extract_features(images)

        self.query(
            points=points, calibs=calibs, transforms=transforms, labels=labels)

        if surpoint is not None:
            rgb = self.query_rgb(
                points=surpoint, calibs=calibs, transforms=transforms)
        else:
            rgb = None
        res = self.preds

        return res, rgb

# The implementation is adopted from OSTrack,
# made publicly available under the MIT License at https://github.com/botaoye/OSTrack/
import torch
import torch.nn as nn


def conv(in_planes,
         out_planes,
         kernel_size=3,
         stride=1,
         padding=1,
         dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class CenterPredictor(
        nn.Module, ):

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel)
        self.conv2_ctr = conv(channel, channel // 2)
        self.conv3_ctr = conv(channel // 2, channel // 4)
        self.conv4_ctr = conv(channel // 4, channel // 8)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # offset regress
        self.conv1_offset = conv(inplanes, channel)
        self.conv2_offset = conv(channel, channel // 2)
        self.conv3_offset = conv(channel // 2, channel // 4)
        self.conv4_offset = conv(channel // 4, channel // 8)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel)
        self.conv2_size = conv(channel, channel // 2)
        self.conv3_size = conv(channel // 2, channel // 4)
        self.conv4_size = conv(channel // 4, channel // 8)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_score=False):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        if return_score:
            bbox, max_score = self.cal_bbox(
                score_map_ctr, size_map, offset_map, return_score=True)
            return score_map_ctr, bbox, size_map, offset_map, max_score
        else:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
            return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self,
                 score_map_ctr,
                 size_map,
                 offset_map,
                 return_score=False):
        max_score, idx = torch.max(
            score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # cx, cy, w, h
        bbox = torch.cat(
            [(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
             (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
             size.squeeze(-1)],
            dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(
            score_map_size), score_map_offset


def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if cfg.MODEL.HEAD.TYPE == 'CENTER':
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(
            inplanes=in_channel,
            channel=out_channel,
            feat_sz=feat_sz,
            stride=stride)
        return center_head
    else:
        raise ValueError('HEAD TYPE %s is not supported.'
                         % cfg.MODEL.HEAD_TYPE)

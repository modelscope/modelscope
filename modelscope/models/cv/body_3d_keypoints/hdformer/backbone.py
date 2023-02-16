# --------------------------------------------------------
# The implementation is also open-sourced by the authors as Hanyuan Chen, and is available publicly on
# https://github.com/hyer/HDFormer
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.body_3d_keypoints.hdformer.block import \
    HightOrderAttentionBlock
from modelscope.models.cv.body_3d_keypoints.hdformer.directed_graph import (
    DiGraph, Graph)
from modelscope.models.cv.body_3d_keypoints.hdformer.skeleton import \
    get_skeleton


class HDFormerNet(nn.Module):

    def __init__(self, cfg):
        super(HDFormerNet, self).__init__()
        in_channels = cfg.in_channels
        dropout = cfg.dropout
        self.cfg = cfg
        self.PLANES = [16, 32, 64, 128, 256]

        # load graph
        skeleton = get_skeleton()
        self.di_graph = DiGraph(skeleton=skeleton)
        self.graph = Graph(
            skeleton=skeleton, strategy='agcn', max_hop=1, dilation=1)
        self.A = torch.tensor(
            self.graph.A,
            dtype=torch.float32,
            requires_grad=True,
            device='cuda')

        # build networks
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        if not cfg.data_bn:
            self.data_bn = None
        else:
            n_joints = self.cfg.IN_NUM_JOINTS \
                if hasattr(self.cfg, 'IN_NUM_JOINTS') \
                else self.cfg.n_joints
            self.data_bn = nn.BatchNorm1d(in_channels * n_joints) if hasattr(cfg, 'PJN') and cfg.PJN \
                else nn.BatchNorm2d(in_channels)

        self.downsample = nn.ModuleList(
            (
                HightOrderAttentionBlock(
                    in_channels,
                    self.PLANES[0],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    residual=False,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=0),
                HightOrderAttentionBlock(
                    self.PLANES[0],
                    self.PLANES[1],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    stride=2,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=0),
                HightOrderAttentionBlock(
                    self.PLANES[1],
                    self.PLANES[1],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=0),
                HightOrderAttentionBlock(
                    self.PLANES[1],
                    self.PLANES[2],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    stride=2,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=0),
                HightOrderAttentionBlock(
                    self.PLANES[2],
                    self.PLANES[2],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=0),
                HightOrderAttentionBlock(
                    self.PLANES[2],
                    self.PLANES[3],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    stride=2,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=dropout),
                HightOrderAttentionBlock(
                    self.PLANES[3],
                    self.PLANES[3],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=dropout),
                HightOrderAttentionBlock(
                    self.PLANES[3],
                    self.PLANES[4],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    stride=2,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=dropout),
                HightOrderAttentionBlock(
                    self.PLANES[4],
                    self.PLANES[4],
                    kernel_size,
                    A=self.A,
                    di_graph=self.di_graph,
                    adj_len=self.A.size(1),
                    attention=cfg.attention_down if hasattr(
                        cfg, 'attention_down') else False,
                    dropout=dropout),
            ))

        self.upsample = nn.ModuleList((
            HightOrderAttentionBlock(
                self.PLANES[4],
                self.PLANES[3],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_up
                if hasattr(cfg, 'attention_up') else False,
                adj_len=self.A.size(1),
                dropout=dropout),
            HightOrderAttentionBlock(
                self.PLANES[3],
                self.PLANES[2],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_up
                if hasattr(cfg, 'attention_up') else False,
                adj_len=self.A.size(1),
                dropout=dropout),
            HightOrderAttentionBlock(
                self.PLANES[2],
                self.PLANES[1],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_up
                if hasattr(cfg, 'attention_up') else False,
                adj_len=self.A.size(1),
                dropout=0),
            HightOrderAttentionBlock(
                self.PLANES[1],
                self.PLANES[0],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_up
                if hasattr(cfg, 'attention_up') else False,
                adj_len=self.A.size(1),
                dropout=0),
        ))

        self.merge = nn.ModuleList((
            HightOrderAttentionBlock(
                self.PLANES[4],
                self.PLANES[0],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_merge if hasattr(
                    cfg, 'attention_merge') else False,
                adj_len=self.A.size(1),
                dropout=dropout,
                max_hop=self.cfg.max_hop),
            HightOrderAttentionBlock(
                self.PLANES[3],
                self.PLANES[0],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_merge if hasattr(
                    cfg, 'attention_merge') else False,
                adj_len=self.A.size(1),
                dropout=dropout,
                max_hop=self.cfg.max_hop),
            HightOrderAttentionBlock(
                self.PLANES[2],
                self.PLANES[0],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_merge if hasattr(
                    cfg, 'attention_merge') else False,
                adj_len=self.A.size(1),
                dropout=0,
                max_hop=self.cfg.max_hop),
            HightOrderAttentionBlock(
                self.PLANES[1],
                self.PLANES[0],
                kernel_size,
                A=self.A,
                di_graph=self.di_graph,
                attention=cfg.attention_merge if hasattr(
                    cfg, 'attention_merge') else False,
                adj_len=self.A.size(1),
                dropout=0,
                max_hop=self.cfg.max_hop),
        ))

    def get_edge_fea(self, x_v):
        x_e = (x_v[..., [c for p, c in self.di_graph.directed_edges_hop1]]
               - x_v[..., [p for p, c in self.di_graph.directed_edges_hop1]]
               ).contiguous()
        N, C, T, V = x_v.shape
        edeg_append = torch.zeros((N, C, T, 1), device=x_e.device)
        x_e = torch.cat((x_e, edeg_append), dim=-1)
        return x_e

    def forward(self, x_v: torch.Tensor):
        """
        x: shape [B,C,T,V_v]
        """
        B, C, T, V = x_v.shape
        # data normalization
        if self.data_bn is not None:
            if hasattr(self.cfg, 'PJN') and self.cfg.PJN:
                x_v = self.data_bn(x_v.permute(0, 1, 3, 2).contiguous().view(B, -1, T)).view(B, C, V, T) \
                    .contiguous().permute(0, 1, 3, 2)
            else:
                x_v = self.data_bn(x_v)

        x_e = self.get_edge_fea(x_v)

        # forward
        feature = []
        for idx, hoa_block in enumerate(self.downsample):
            x_v, x_e = hoa_block(x_v, x_e)
            if idx == 0 or idx == 2 or idx == 4 or idx == 6:
                feature.append((x_v, x_e))

        feature.append((x_v, x_e))
        feature = feature[::-1]

        x_v, x_e = feature[0]
        identity_feature = feature[1:]

        ushape_feature = []
        ushape_feature.append((x_v, x_e))
        for idx, (hoa_block, id) in \
                enumerate(zip(self.upsample, identity_feature)):
            x_v, x_e = hoa_block(x_v, x_e)
            if hasattr(self.cfg, 'deterministic') and self.cfg.deterministic:
                x_v = F.interpolate(x_v, scale_factor=(2, 1), mode='nearest')
            else:
                x_v = F.interpolate(
                    x_v,
                    scale_factor=(2, 1),
                    mode='bilinear',
                    align_corners=False)
            x_v += id[0]
            ushape_feature.append((x_v, x_e))

        ushape_feature = ushape_feature[:-1]
        for idx, (hoa_block, u) in \
                enumerate(zip(self.merge, ushape_feature)):
            x_v2, x_e2 = hoa_block(*u)
            if hasattr(self.cfg, 'deterministic') and self.cfg.deterministic:
                x_v += F.interpolate(
                    x_v2, scale_factor=(2**(4 - idx), 1), mode='nearest')
            else:
                x_v += F.interpolate(
                    x_v2,
                    scale_factor=(2**(4 - idx), 1),
                    mode='bilinear',
                    align_corners=False)
        return x_v, x_e

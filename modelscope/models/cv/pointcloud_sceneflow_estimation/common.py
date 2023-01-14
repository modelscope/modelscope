# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils as pointutils

RADIUS = 2.5


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped,
                                               knn_idx.int()).permute(
                                                   0, 2, 3, 1)

    return new_points


def curvature(pc, nsample=10, radius=RADIUS):
    # pc: B 3 N
    assert pc.shape[1] == 3
    pc = pc.permute(0, 2, 1)

    dist, kidx = pointutils.knn(nsample, pc.contiguous(),
                                pc.contiguous())  # (B, N, 10)

    if radius is not None:
        tmp_idx = kidx[:, :, 0].unsqueeze(2).repeat(1, 1,
                                                    nsample).to(kidx.device)
        kidx[dist > radius] = tmp_idx[dist > radius]

    grouped_pc = index_points_group(pc, kidx)  # B N 10 3
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim=2) / 9.0
    return pc_curvature  # B N 3


class PointNetSetAbstractionRatio(nn.Module):

    def __init__(self,
                 ratio,
                 radius,
                 nsample,
                 in_channel,
                 mlp,
                 group_all,
                 return_fps=False,
                 use_xyz=True,
                 use_act=True,
                 act=F.relu,
                 mean_aggr=False,
                 use_instance_norm=False):
        super(PointNetSetAbstractionRatio, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        self.act = act
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if use_instance_norm:
                self.mlp_bns.append(
                    nn.InstanceNorm2d(out_channel, affine=True))
            else:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if group_all:
            self.queryandgroup = pointutils.GroupAll(self.use_xyz)
        else:
            self.queryandgroup = pointutils.QueryAndGroup(
                radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape
        npoint = int(N * self.ratio)

        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        if (self.group_all is False) and (npoint != -1):
            if fps_idx is None:
                fps_idx = pointutils.furthest_point_sample(xyz_t,
                                                           npoint)  # [B, N]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        else:
            new_xyz = xyz
        new_points, _ = self.queryandgroup(xyz_t,
                                           new_xyz.transpose(2,
                                                             1).contiguous(),
                                           points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_act:
                bn = self.mlp_bns[i]
                new_points = self.act(bn(conv(new_points)))
            else:
                new_points = conv(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1)
        else:
            new_points = torch.max(new_points, -1)[0]

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):

    def __init__(self,
                 npoint,
                 radius,
                 nsample,
                 in_channel,
                 mlp,
                 group_all,
                 return_fps=False,
                 use_xyz=True,
                 use_act=True,
                 act=F.relu,
                 mean_aggr=False,
                 use_instance_norm=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        self.act = act
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if use_instance_norm:
                self.mlp_bns.append(
                    nn.InstanceNorm2d(out_channel, affine=True))
            else:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if group_all:
            self.queryandgroup = pointutils.GroupAll(self.use_xyz)
        else:
            self.queryandgroup = pointutils.QueryAndGroup(
                radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        # device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        if (self.group_all is False) and (self.npoint != -1):
            if fps_idx is None:
                fps_idx = pointutils.furthest_point_sample(
                    xyz_t, self.npoint)  # [B, N]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        else:
            new_xyz = xyz
        new_points, _ = self.queryandgroup(xyz_t,
                                           new_xyz.transpose(2,
                                                             1).contiguous(),
                                           points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_act:
                bn = self.mlp_bns[i]
                new_points = self.act(bn(conv(new_points)))
            else:
                new_points = conv(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1)
        else:
            new_points = torch.max(new_points, -1)[0]

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points


class PointNetFeaturePropogation(nn.Module):

    def __init__(self, in_channel, mlp, learn_mask=False, nsample=3):
        super(PointNetFeaturePropogation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.apply_mlp = mlp is not None
        last_channel = in_channel
        self.nsample = nsample
        if self.apply_mlp:
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel

        if learn_mask:
            self.queryandgroup = pointutils.QueryAndGroup(
                None, 9, use_xyz=True)
            last_channel = (128 + 3)
            for out_channel in [32, 1]:
                self.mlp_convs.append(
                    nn.Conv2d(last_channel, out_channel, 1, bias=False))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2, hidden=None):
        """
        Input:
            pos1: input points position data, [B, C, N]
            pos2: sampled input points position data, [B, C, S]
            feature1: input points data, [B, D, N]
            feature2: input points data, [B, D, S]
        Return:
            feat_new: upsampled points data, [B, D', N]
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape

        if hidden is None:
            if self.nsample == 3:
                dists, idx = pointutils.three_nn(pos1_t, pos2_t)
            else:
                dists, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists
            weight = weight / torch.sum(weight, -1, keepdim=True)  # [B,N,3]
            interpolated_feat = torch.sum(
                pointutils.grouping_operation(feature2, idx)
                * weight.view(B, 1, N, self.nsample),
                dim=-1)  # [B,C,N,3]
        else:
            dist, idx = pointutils.knn(9, pos1_t, pos2_t)

            new_feat, _ = self.queryandgroup(pos2_t, pos1_t,
                                             hidden)  # [B, 3+C, N, 9]

            for i, conv in enumerate(self.mlp_convs):
                new_feat = conv(new_feat)
            weight = torch.softmax(new_feat, dim=-1)  # [B, 1, N, 9]
            interpolated_feat = torch.sum(
                pointutils.grouping_operation(feature2, idx) * weight,
                dim=-1)  # [B, C, N]

        if feature1 is not None:
            feat_new = torch.cat([interpolated_feat, feature1], 1)
        else:
            feat_new = interpolated_feat

        if self.apply_mlp:
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                feat_new = F.relu(bn(conv(feat_new)))
        return feat_new


class Sinkhorn(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, corr, epsilon, gamma, max_iter):
        # Early return if no iteration
        if max_iter == 0:
            return corr

        # Init. of Sinkhorn algorithm
        power = gamma / (gamma + epsilon)
        a = (
            torch.ones((corr.shape[0], corr.shape[1], 1),
                       device=corr.device,
                       dtype=corr.dtype) / corr.shape[1])
        prob1 = (
            torch.ones((corr.shape[0], corr.shape[1], 1),
                       device=corr.device,
                       dtype=corr.dtype) / corr.shape[1])
        prob2 = (
            torch.ones((corr.shape[0], corr.shape[2], 1),
                       device=corr.device,
                       dtype=corr.dtype) / corr.shape[2])

        # Sinkhorn algorithm
        for _ in range(max_iter):
            # Update b
            KTa = torch.bmm(corr.transpose(1, 2), a)
            b = torch.pow(prob2 / (KTa + 1e-8), power)
            # Update a
            Kb = torch.bmm(corr, b)
            a = torch.pow(prob1 / (Kb + 1e-8), power)

        # Transportation map
        T = torch.mul(torch.mul(a, corr), b.transpose(1, 2))

        return T


class PointWiseOptimLayer(nn.Module):

    def __init__(self, nsample, radius, in_channel, mlp, use_curvature=True):
        super().__init__()
        self.nsample = nsample
        self.radius = radius
        self.use_curvature = use_curvature

        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.ReLU(inplace=True), nn.Conv1d(32, 64, 1))

        self.qk_net = nn.Sequential(
            nn.Conv1d(in_channel + 64, in_channel + 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel + 64, in_channel + 64, 1))
        if self.use_curvature:
            self.curvate_net = nn.Sequential(
                nn.Conv1d(3, 32, 1), nn.ReLU(inplace=True),
                nn.Conv1d(32, 32, 1))
            self.mlp_conv = nn.Conv1d(
                in_channel + 64 + 32, mlp[-1], 1, bias=True)
        else:
            self.mlp_conv = nn.Conv1d(in_channel + 64, mlp[-1], 1, bias=True)

    def forward(self,
                pos1,
                pos2,
                feature1,
                feature2,
                nsample,
                radius=None,
                pos1_raw=None,
                return_score=False):
        """
        Input:
            pos1: (batch_size, 3, npoint)
            pos2: (batch_size, 3, npoint)
            feature1: (batch_size, channel, npoint)
            feature2: (batch_size, channel, npoint)
        Output:
            pos1: (batch_size, 3, npoint)
            cost: (batch_size, channel, npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        self.nsample = nsample
        self.radius = radius

        dist, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)  # [B, N, K]
        if self.radius is not None:
            tmp_idx = idx[:, :,
                          0].unsqueeze(2).repeat(1, 1,
                                                 self.nsample).to(idx.device)
            idx[dist > self.radius] = tmp_idx[dist > self.radius]

        pos1_embed_norm = self.pos_embed(pos1)
        pos2_embed_norm = self.pos_embed(pos2)  # [B, C1, N]

        feat1_w_pos = torch.cat([feature1, pos1_embed_norm], dim=1)
        feat2_w_pos = torch.cat([feature2, pos2_embed_norm],
                                dim=1)  # [B, C1+C2, N]

        feat1_w_pos = self.qk_net(feat1_w_pos)
        feat2_w_pos = self.qk_net(feat2_w_pos)  # [B, C1+C2, N]

        feat2_grouped = pointutils.grouping_operation(feat2_w_pos,
                                                      idx)  # [B, C1+C2, N, S]

        score = torch.softmax(
            feat1_w_pos.unsqueeze(-1) * feat2_grouped * 1.
            / math.sqrt(feat1_w_pos.shape[1]),
            dim=-1)  # [B, C1+C2, N, S]
        cost = (score * (feat1_w_pos.unsqueeze(-1) - feat2_grouped)**2).sum(
            dim=-1)  # [B, C1+C2, N]

        if self.use_curvature:
            curvate1_raw = curvature(pos1_raw).permute(0, 2, 1)  # [B, 3, N]
            curvate1 = curvature(pos1).permute(0, 2, 1)  # [B, 3, N]
            curvate_cost = self.curvate_net(curvate1_raw) - self.curvate_net(
                curvate1)
            curvate_cost = curvate_cost**2
            cost = self.mlp_conv(torch.cat([cost, curvate_cost],
                                           dim=1))  # [B, C, N]
        else:
            cost = self.mlp_conv(cost)  # [B, C, N]

        if return_score:
            pos2_grouped = pointutils.grouping_operation(pos2,
                                                         idx)  # [B, 3, N, S]
            # [B, N, K]
            index = (dist > self.radius).sum(
                dim=2, keepdim=True).float() > (dist.shape[2] - 0.1
                                                )  # [B, N, 1]
            index = index.unsqueeze(1).repeat(1, score.shape[1], 1,
                                              dist.shape[2])  # [B, N, K]
            score_tmp = score.clone()
            score_tmp[index] = 0.0
            score = score_tmp
            return pos1, cost, score, pos2_grouped
        else:
            return pos1, cost

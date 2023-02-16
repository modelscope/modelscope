# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (PointNetFeaturePropogation, PointNetSetAbstraction,
                     PointWiseOptimLayer, Sinkhorn)


class FeatureMatching(nn.Module):

    def __init__(self, npoint, use_instance_norm, supporth_th, feature_norm,
                 max_iter):
        super(FeatureMatching, self).__init__()
        self.support_th = supporth_th**2  # 10m
        self.feature_norm = feature_norm
        self.max_iter = max_iter
        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        # Entropic regularisation
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.sinkhorn = Sinkhorn()

        self.extract_glob = FeatureExtractionGlobal(npoint, use_instance_norm)

        # upsample flow
        self.fp0 = PointNetFeaturePropogation(in_channel=3, mlp=[])
        self.sa1 = PointNetSetAbstraction(
            npoint=int(npoint / 16),
            radius=None,
            nsample=16,
            in_channel=3,
            mlp=[32, 32, 64],
            group_all=False,
            use_instance_norm=use_instance_norm)
        self.fp1 = PointNetFeaturePropogation(in_channel=64, mlp=[])
        self.sa2 = PointNetSetAbstraction(
            npoint=int(npoint / 8),
            radius=None,
            nsample=16,
            in_channel=64,
            mlp=[64, 64, 128],
            group_all=False,
            use_instance_norm=use_instance_norm)
        self.fp2 = PointNetFeaturePropogation(in_channel=128, mlp=[])

        self.flow_regressor = FlowRegressor(npoint, use_instance_norm)
        self.flow_up_sample = PointNetFeaturePropogation(in_channel=3, mlp=[])

    def upsample_flow(self, pc1_l, pc1_l_glob, flow_inp):
        """
            flow_inp: [B, N, 3]
            return: [B, 3, N]
        """

        flow_inp = flow_inp.permute(0, 2, 1).contiguous()  # [B, 3, N]

        flow_feat = self.fp0(pc1_l_glob['s16'], pc1_l_glob['s32'], None,
                             flow_inp)
        _, corr_feats_l2 = self.sa1(pc1_l_glob['s16'], flow_feat)

        flow_feat = self.fp1(pc1_l_glob['s8'], pc1_l_glob['s16'], None,
                             corr_feats_l2)
        _, flow_feat = self.sa2(pc1_l_glob['s8'], flow_feat)

        flow_feat = self.fp2(pc1_l['s4'], pc1_l_glob['s8'], None, flow_feat)

        flow, flow_lr = self.flow_regressor(pc1_l, flow_feat)

        flow_up = self.flow_up_sample(pc1_l['s1'], pc1_l_glob['s32'], None,
                                      flow_inp)
        flow_lr_up = self.flow_up_sample(pc1_l['s4'], pc1_l_glob['s32'], None,
                                         flow_inp)

        flow, flow_lr = flow + flow_up, flow_lr + flow_lr_up

        return flow, flow_lr

    def calc_feats_corr(self, pcloud1, pcloud2, feature1, feature2, norm):
        """
            pcloud1, pcloud2: [B, N, 3]
            feature1, feature2: [B, N, C]
        """
        if norm:
            feature1 = feature1 / torch.sqrt(
                torch.sum(feature1**2, -1, keepdim=True) + 1e-6)
            feature2 = feature2 / torch.sqrt(
                torch.sum(feature2**2, -1, keepdim=True) + 1e-6)
            corr_mat = torch.bmm(feature1,
                                 feature2.transpose(1, 2))  # [B, N1, N2]
        else:
            corr_mat = torch.bmm(feature1, feature2.transpose(
                1, 2)) / feature1.shape[2]**.5  # [B, N1, N2]

        if self.support_th is not None:
            distance_matrix = torch.sum(
                pcloud1**2, -1, keepdim=True)  # [B, N1, 1]
            distance_matrix = distance_matrix + torch.sum(
                pcloud2**2, -1, keepdim=True).transpose(1, 2)  # [B, N1, N2]
            distance_matrix = distance_matrix - 2 * torch.bmm(
                pcloud1, pcloud2.transpose(1, 2))  # [B, N1, N2]
            support = (distance_matrix < self.support_th)  # [B, N1, N2]
            support = support.float()
        else:
            support = torch.ones_like(corr_mat)
        return corr_mat, support

    def calc_corr_mat(self, pcloud1, pcloud2, feature1, feature2):
        """
            pcloud1, pcloud2: [B, N, 3]
            feature1, feature2: [B, N, C]
            corr_mat: [B, N1, N2]
        """
        epsilon = torch.exp(self.epsilon) + 0.03
        corr_mat, support = self.calc_feats_corr(
            pcloud1, pcloud2, feature1, feature2, norm=self.feature_norm)
        C = 1.0 - corr_mat
        corr_mat = torch.exp(-C / epsilon) * support
        return corr_mat

    def get_flow_init(self, pcloud1, pcloud2, feats1, feats2):
        """
            pcloud1, pcloud2: [B, 3, N]
            feats1, feats2: [B, C, N]
        """

        corr_mat = self.calc_corr_mat(
            pcloud1.permute(0, 2, 1), pcloud2.permute(0, 2, 1),
            feats1.permute(0, 2, 1), feats2.permute(0, 2, 1))

        corr_mat = self.sinkhorn(corr_mat,
                                 torch.exp(self.epsilon) + 0.03, self.gamma,
                                 self.max_iter)

        row_sum = corr_mat.sum(-1, keepdim=True)  # [B, N1, 1]
        flow_init = (corr_mat @ pcloud2.permute(0, 2, 1).contiguous()) / (
            row_sum + 1e-6) - pcloud1.permute(0, 2,
                                              1).contiguous()  # [B, N1, 3]

        return flow_init

    def forward(self, pc1_l, pc2_l, feats1, feats2):
        """
        pc1_l, pc2_l: dict([B, 3, N])
        feats1, feats2: [B, C, N]
        """
        pc1_l_glob, feats1_glob = self.extract_glob(pc1_l['s4'], feats1)
        pc2_l_glob, feats2_glob = self.extract_glob(pc2_l['s4'], feats2)

        flow_init_s32 = self.get_flow_init(pc1_l_glob['s32'],
                                           pc2_l_glob['s32'], feats1_glob,
                                           feats2_glob)

        flow_init, flow_init_s4 = self.upsample_flow(pc1_l, pc1_l_glob,
                                                     flow_init_s32)

        return flow_init, flow_init_s4


class FlowRegressor(nn.Module):

    def __init__(self, npoint, use_instance_norm, input_dim=128, nsample=32):
        super(FlowRegressor, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=int(npoint / 4),
            radius=None,
            nsample=nsample,
            in_channel=input_dim,
            mlp=[input_dim, input_dim],
            group_all=False,
            use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(
            npoint=int(npoint / 4),
            radius=None,
            nsample=nsample,
            in_channel=input_dim,
            mlp=[input_dim, input_dim],
            group_all=False,
            use_instance_norm=use_instance_norm)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(inplace=True),
            nn.Linear(input_dim, 3))

        self.up_sample = PointNetFeaturePropogation(in_channel=3, mlp=[])

    def forward(self, pc1_l, feats):
        """
            pc1_l: dict([B, 3, N])
            feats: [B, C, N]
            return: [B, 3, N]
        """
        _, x = self.sa1(pc1_l['s4'], feats)
        _, x = self.sa2(pc1_l['s4'], x)
        x = x.permute(0, 2, 1).contiguous()  # [B, N, C]
        x = self.fc(x)
        flow_lr = x.permute(0, 2, 1).contiguous()  # [B, 3, N]

        flow = self.up_sample(pc1_l['s1'], pc1_l['s4'], None,
                              flow_lr)  # [B, 3, N]

        return flow, flow_lr


class FeatureExtractionGlobal(nn.Module):

    def __init__(self, npoint, use_instance_norm):
        super(FeatureExtractionGlobal, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=int(npoint / 8),
            radius=None,
            nsample=32,
            in_channel=64,
            mlp=[128, 128, 128],
            group_all=False,
            use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(
            npoint=int(npoint / 16),
            radius=None,
            nsample=24,
            in_channel=128,
            mlp=[128, 128, 128],
            group_all=False,
            use_instance_norm=use_instance_norm)
        self.sa3 = PointNetSetAbstraction(
            npoint=int(npoint / 32),
            radius=None,
            nsample=16,
            in_channel=128,
            mlp=[256, 256, 256],
            group_all=False,
            use_instance_norm=use_instance_norm)

    def forward(self, pc, feature):
        pc_l1, feat_l1 = self.sa1(pc, feature)
        pc_l2, feat_l2 = self.sa2(pc_l1, feat_l1)
        pc_l3, feat_l3 = self.sa3(pc_l2, feat_l2)

        pc_l = dict(s8=pc_l1, s16=pc_l2, s32=pc_l3)
        return pc_l, feat_l3


class FeatureExtraction(nn.Module):

    def __init__(self, npoint, use_instance_norm):
        super(FeatureExtraction, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=int(npoint / 2),
            radius=None,
            nsample=32,
            in_channel=3,
            mlp=[32, 32, 32],
            group_all=False,
            return_fps=True,
            use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(
            npoint=int(npoint / 4),
            radius=None,
            nsample=32,
            in_channel=32,
            mlp=[64, 64, 64],
            group_all=False,
            return_fps=True,
            use_instance_norm=use_instance_norm)

    def forward(self, pc, feature, fps_idx=None):
        """
            pc: [B, 3, N]
            feature: [B, 3, N]
        """
        fps_idx1 = fps_idx['s2'] if fps_idx is not None else None
        pc_l1, feat_l1, fps_idx1 = self.sa1(pc, feature, fps_idx=fps_idx1)
        fps_idx2 = fps_idx['s4'] if fps_idx is not None else None
        pc_l2, feat_l2, fps_idx2 = self.sa2(pc_l1, feat_l1, fps_idx=fps_idx2)
        pc_l = dict(s1=pc, s2=pc_l1, s4=pc_l2)
        fps_idx = dict(s2=fps_idx1, s4=fps_idx2)
        return pc_l, feat_l2, fps_idx


class HiddenInitNet(nn.Module):

    def __init__(self, npoint, use_instance_norm):
        super(HiddenInitNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=int(npoint / 4),
            radius=None,
            nsample=8,
            in_channel=64,
            mlp=[128, 128, 128],
            group_all=False,
            use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(
            npoint=int(npoint / 4),
            radius=None,
            nsample=8,
            in_channel=128,
            mlp=[128],
            group_all=False,
            use_act=False,
            use_instance_norm=use_instance_norm)

    def forward(self, pc, feature):
        _, feat_l1 = self.sa1(pc, feature)
        _, feat_l2 = self.sa2(pc, feat_l1)

        h_init = torch.tanh(feat_l2)
        return h_init


class GRUReg(nn.Module):

    def __init__(self, npoint, hidden_dim, input_dim, use_instance_norm):
        super().__init__()
        in_ch = hidden_dim + input_dim

        self.flow_proj = nn.ModuleList([
            PointNetSetAbstraction(
                npoint=int(npoint / 4),
                radius=None,
                nsample=16,
                in_channel=3,
                mlp=[32, 32, 32],
                group_all=False,
                use_instance_norm=use_instance_norm),
            PointNetSetAbstraction(
                npoint=int(npoint / 4),
                radius=None,
                nsample=8,
                in_channel=32,
                mlp=[16, 16, 16],
                group_all=False,
                use_instance_norm=use_instance_norm)
        ])

        self.hidden_init_net = HiddenInitNet(npoint, use_instance_norm)

        self.gru_layers = nn.ModuleList([
            PointNetSetAbstraction(
                npoint=int(npoint / 4),
                radius=None,
                nsample=4,
                in_channel=in_ch,
                mlp=[hidden_dim],
                group_all=False,
                use_act=False,
                use_instance_norm=use_instance_norm),
            PointNetSetAbstraction(
                npoint=int(npoint / 4),
                radius=None,
                nsample=4,
                in_channel=in_ch,
                mlp=[hidden_dim],
                group_all=False,
                use_act=False,
                use_instance_norm=use_instance_norm),
            PointNetSetAbstraction(
                npoint=int(npoint / 4),
                radius=None,
                nsample=4,
                in_channel=in_ch,
                mlp=[hidden_dim],
                group_all=False,
                use_act=False,
                use_instance_norm=use_instance_norm)
        ])

    def gru(self, h, gru_inp, pc):
        hx = torch.cat([h, gru_inp], dim=1)
        z = torch.sigmoid(self.gru_layers[0](pc, hx)[1])
        r = torch.sigmoid(self.gru_layers[1](pc, hx)[1])
        q = torch.tanh(self.gru_layers[2](pc, torch.cat([r * h, gru_inp],
                                                        dim=1))[1])
        h = (1 - z) * h + z * q
        return h

    def get_gru_input(self, feats1_new, cost, flow, pc):
        flow_feats = flow
        for flow_conv in self.flow_proj:
            _, flow_feats = flow_conv(pc, flow_feats)

        gru_inp = torch.cat([feats1_new, cost, flow_feats, flow],
                            dim=1)  # [64, 128, 16, 3]

        return gru_inp

    def forward(self, h, feats1_new, cost, flow_lr, pc1_l):
        gru_inp = self.get_gru_input(feats1_new, cost, flow_lr, pc=pc1_l['s4'])

        h = self.gru(h, gru_inp, pc1_l['s4'])
        return h


class SF_RCP(nn.Module):

    def __init__(self, npoint=8192, use_instance_norm=False, **kwargs):
        super().__init__()
        self.radius = kwargs.get('radius', 3.5)
        self.nsample = kwargs.get('nsample', 6)
        self.radius_min = kwargs.get('radius_min', 3.5)
        self.nsample_min = kwargs.get('nsample_min', 6)
        self.use_curvature = kwargs.get('use_curvature', True)
        self.flow_ratio = kwargs.get('flow_ratio', 0.1)
        self.init_max_iter = kwargs.get('init_max_iter', 0)
        self.init_feature_norm = kwargs.get('init_feature_norm', True)
        self.support_th = kwargs.get('support_th', 10)

        self.feature_extraction = FeatureExtraction(npoint, use_instance_norm)
        self.feature_matching = FeatureMatching(
            npoint,
            use_instance_norm,
            supporth_th=self.support_th,
            feature_norm=self.init_feature_norm,
            max_iter=self.init_max_iter)

        self.pointwise_optim_layer = PointWiseOptimLayer(
            nsample=self.nsample,
            radius=self.radius,
            in_channel=64,
            mlp=[128, 128, 128],
            use_curvature=self.use_curvature)

        self.gru = GRUReg(
            npoint,
            hidden_dim=128,
            input_dim=128 + 64 + 16 + 3,
            use_instance_norm=use_instance_norm)

        self.flow_regressor = FlowRegressor(npoint, use_instance_norm)

    def initialization(self, pc1_l, pc2_l, feats1, feats2):
        """
            pc1: [B, 3, N]
            pc2: [B, 3, N]
            feature1: [B, 3, N]
            feature2: [B, 3, N]
        """
        flow, flow_lr = self.feature_matching(pc1_l, pc2_l, feats1, feats2)

        return flow, flow_lr

    def pointwise_optimization(self, pc1_l_new, pc2_l, feats1_new, feats2,
                               pc1_l, flow_lr, iter):
        _, cost, score, pos2_grouped = self.pointwise_optim_layer(
            pc1_l_new['s4'],
            pc2_l['s4'],
            feats1_new,
            feats2,
            nsample=max(self.nsample_min, self.nsample // (2**iter)),
            radius=max(self.radius_min, self.radius / (2**iter)),
            pos1_raw=pc1_l['s4'],
            return_score=True)

        # pc1_new_l_loc: [B, 3, N, S]
        # pos2_grouped: [B, C, N, S]
        delta_flow_tmp = ((pos2_grouped - pc1_l_new['s4'].unsqueeze(-1))
                          * score.mean(dim=1, keepdim=True)).sum(
                              dim=-1)  # [B, 3, N]
        flow_lr = flow_lr + self.flow_ratio * delta_flow_tmp

        return flow_lr, cost

    def update_pos(self, pc, pc_lr, flow, flow_lr):
        pc = pc + flow
        pc_lr = pc_lr + flow_lr
        return pc, pc_lr

    def forward(self, pc1, pc2, feature1, feature2, iters=1):
        """
            pc1: [B, N, 3]
            pc2: [B, N, 3]
            feature1: [B, N, 3]
            feature2: [B, N, 3]
        """
        # prepare
        flow_predictions = []
        pc1 = pc1.permute(0, 2, 1).contiguous()  # B 3 N
        pc2 = pc2.permute(0, 2, 1).contiguous()  # B 3 N
        feature1 = feature1.permute(0, 2, 1).contiguous()  # B 3 N
        feature2 = feature2.permute(0, 2, 1).contiguous()  # B 3 N

        # feature extraction
        pc1_l, feats1, fps_idx1 = self.feature_extraction(pc1, feature1)
        pc2_l, feats2, _ = self.feature_extraction(pc2, feature2)

        # initialization, flow_lr_init(flow_low_resolution)
        flow_init, flow_lr_init = self.initialization(pc1_l, pc2_l, feats1,
                                                      feats2)
        flow_predictions.append(flow_init.permute(0, 2, 1))

        # gru init hidden state
        h = self.gru.hidden_init_net(pc1_l['s4'], feats1)

        # update position
        pc1_lr_raw = pc1_l['s4']
        pc1_new, pc1_lr_new = self.update_pos(pc1, pc1_lr_raw, flow_init,
                                              flow_lr_init)

        # iterative optim
        for iter in range(iters - 1):
            pc1_new = pc1_new.detach()
            pc1_lr_new = pc1_lr_new.detach()
            flow_lr = pc1_lr_new - pc1_lr_raw

            pc1_l_new, feats1_new, _ = self.feature_extraction(
                pc1_new, pc1_new, fps_idx1)

            # pointwise optimization to get udpated flow_lr and cost
            flow_lr_update, cost = self.pointwise_optimization(
                pc1_l_new, pc2_l, feats1_new, feats2, pc1_l, flow_lr, iter)
            flow_lr = flow_lr_update

            # gru regularization
            h = self.gru(h, feats1_new, cost, flow_lr, pc1_l)
            # pred flow_lr
            delta_flow, delta_flow_lr = self.flow_regressor(pc1_l, h)

            pc1_new, pc1_lr_new = self.update_pos(pc1_new, pc1_lr_new,
                                                  delta_flow, delta_flow_lr)

            flow = pc1_new - pc1
            flow_predictions.append(flow.permute(0, 2, 1))

        return flow_predictions

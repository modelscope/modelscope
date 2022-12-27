# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_d, act_fn=F.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)


class PoseHead(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128):
        super(PoseHead, self).__init__()

        self.conv1_pose = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2_pose = nn.Conv2d(hidden_dim, 6, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_p):
        out = self.conv2_pose(self.relu(self.conv1_pose(x_p))).mean(3).mean(2)
        return torch.cat([out[:, :3], 0.01 * out[:, 3:]], dim=1)


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ProjectionInputDepth(nn.Module):

    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd = nn.Conv2d(64 + hidden_dim, out_chs - 1, 3, padding=1)

    def forward(self, depth, cost):
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)

        out_d = F.relu(self.convd(cor_dfm))

        return torch.cat([out_d, depth], dim=1)


class ProjectionInputPose(nn.Module):

    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.convp1 = nn.Conv2d(6, hidden_dim, 7, padding=3)
        self.convp2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convp = nn.Conv2d(64 + hidden_dim, out_chs - 6, 3, padding=1)

    def forward(self, pose, cost):
        bs, _, h, w = cost.shape
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        pfm = F.relu(self.convp1(pose.view(bs, 6, 1, 1).repeat(1, 1, h, w)))
        pfm = F.relu(self.convp2(pfm))
        cor_pfm = torch.cat([cor, pfm], dim=1)

        out_p = F.relu(self.convp(cor_pfm))
        return torch.cat(
            [out_p, pose.view(bs, 6, 1, 1).repeat(1, 1, h, w)], dim=1)


class UpMaskNet(nn.Module):

    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, ratio * ratio * 9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask


class BasicUpdateBlockDepth(nn.Module):

    def __init__(self, hidden_dim=128, cost_dim=256, ratio=8, context_dim=64):
        super(BasicUpdateBlockDepth, self).__init__()

        self.encoder = ProjectionInputDepth(
            cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.depth_gru = SepConvGRU(
            hidden_dim=hidden_dim,
            input_dim=self.encoder.out_chs + context_dim)
        self.depth_head = DepthHead(
            hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, ratio * ratio * 9, 1, padding=0))

    def forward(self,
                net,
                cost_func,
                inv_depth,
                context,
                seq_len=4,
                scale_func=None):
        inv_depth_list = []
        mask_list = []
        for i in range(seq_len):
            # TODO detach()
            # inv_depth = inv_depth.detach()
            input_features = self.encoder(inv_depth,
                                          cost_func(scale_func(inv_depth)[0]))
            inp_i = torch.cat([context, input_features], dim=1)

            net = self.depth_gru(net, inp_i)
            delta_inv_depth = self.depth_head(net)
            # scale mask to balence gradients
            mask = .25 * self.mask(net)

            inv_depth = inv_depth + delta_inv_depth
            inv_depth_list.append(inv_depth)
            mask_list.append(mask)

        return net, mask_list, inv_depth_list


class BasicUpdateBlockPose(nn.Module):

    def __init__(self, hidden_dim=128, cost_dim=256, context_dim=64):
        super(BasicUpdateBlockPose, self).__init__()
        self.encoder = ProjectionInputPose(
            cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.pose_gru = SepConvGRU(
            hidden_dim=hidden_dim,
            input_dim=self.encoder.out_chs + context_dim)
        self.pose_head = PoseHead(hidden_dim, hidden_dim=hidden_dim)

    def forward(self, net, cost_func, pose, inp, seq_len=4):
        pose_list = []
        for i in range(seq_len):
            # TODO detach()
            # pose = pose.detach()
            input_features = self.encoder(pose, cost_func(pose))
            inp_i = torch.cat([inp, input_features], dim=1)

            net = self.pose_gru(net, inp_i)
            delta_pose = self.pose_head(net)

            pose = pose + delta_pose
            pose_list.append(pose)

        # scale mask to balence gradients
        return net, pose_list

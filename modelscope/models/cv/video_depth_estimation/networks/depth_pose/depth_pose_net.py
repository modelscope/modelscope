# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_depth_estimation.geometry.camera import (
    Camera, Pose)
from modelscope.models.cv.video_depth_estimation.networks.layers.resnet.layers import \
    disp_to_depth
from modelscope.models.cv.video_depth_estimation.networks.optim.extractor import \
    ResNetEncoder
from modelscope.models.cv.video_depth_estimation.networks.optim.update import (
    BasicUpdateBlockDepth, BasicUpdateBlockPose, DepthHead, PoseHead,
    UpMaskNet)
from modelscope.models.cv.video_depth_estimation.utils.depth import inv2depth


class DepthPoseNet(nn.Module):

    def __init__(self, version=None, min_depth=0.1, max_depth=100, **kwargs):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert 'it' in version
        self.iters = int(version.split('-')[0].split('it')[1])
        self.is_high = 'h' in version
        self.out_normalize = 'out' in version
        # get seq len in one stage. default: 4.
        self.seq_len = 4
        for str in version.split('-'):
            if 'seq' in str:
                self.seq_len = int(str.split('seq')[1])
        # update iters
        self.iters = self.iters // self.seq_len
        # intermediate supervision
        self.inter_sup = 'inter' in version

        print(
            f'=======iters:{self.iters}, sub_seq_len:{self.seq_len}, inter_sup: {self.inter_sup}, '
            f'is_high:{self.is_high}, out_norm:{self.out_normalize}, '
            f'max_depth:{self.max_depth} min_depth:{self.min_depth}========')

        if self.out_normalize:
            self.scale_inv_depth = partial(
                disp_to_depth,
                min_depth=self.min_depth,
                max_depth=self.max_depth)
        else:
            self.scale_inv_depth = lambda x: (x, None)  # identity

        # feature network, context network, and update block
        self.foutput_dim = 128
        self.feat_ratio = 8
        self.fnet = ResNetEncoder(
            out_chs=self.foutput_dim, stride=self.feat_ratio)

        self.depth_head = DepthHead(
            input_dim=self.foutput_dim,
            hidden_dim=self.foutput_dim,
            scale=False)
        self.pose_head = PoseHead(
            input_dim=self.foutput_dim * 2, hidden_dim=self.foutput_dim)
        self.upmask_net = UpMaskNet(
            hidden_dim=self.foutput_dim, ratio=self.feat_ratio)

        self.hdim = 128 if self.is_high else 64
        self.cdim = 32

        self.update_block_depth = BasicUpdateBlockDepth(
            hidden_dim=self.hdim,
            cost_dim=self.foutput_dim,
            ratio=self.feat_ratio,
            context_dim=self.cdim)
        self.update_block_pose = BasicUpdateBlockPose(
            hidden_dim=self.hdim,
            cost_dim=self.foutput_dim,
            context_dim=self.cdim)

        self.cnet = ResNetEncoder(
            out_chs=self.foutput_dim, stride=self.feat_ratio)
        self.cnet_depth = ResNetEncoder(
            out_chs=self.hdim + self.cdim,
            stride=self.feat_ratio,
            num_input_images=1)
        self.cnet_pose = ResNetEncoder(
            out_chs=self.hdim + self.cdim,
            stride=self.feat_ratio,
            num_input_images=2)

    def upsample_depth(self, depth, mask, ratio=8):
        """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape
        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(depth, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, ratio * H, ratio * W)

    def get_cost_each(self, pose, fmap, fmap_ref, depth, K, ref_K,
                      scale_factor):
        """
            depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        pose = Pose.from_vec(pose, 'euler')

        device = depth.device
        cam = Camera(K=K.float()).scaled(scale_factor).to(
            device)  # tcw = Identity
        ref_cam = Camera(
            K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device)

        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(
            world_points, frame='w', normalize=True)  # (b, h, w,2)

        fmap_warped = F.grid_sample(
            fmap_ref,
            ref_coords,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)  # (b, c, h, w)

        cost = (fmap - fmap_warped)**2

        return cost

    def depth_cost_calc(self, inv_depth, fmap, fmaps_ref, pose_list, K, ref_K,
                        scale_factor):
        cost_list = []
        for pose, fmap_r in zip(pose_list, fmaps_ref):
            cost = self.get_cost_each(pose, fmap, fmap_r, inv2depth(inv_depth),
                                      K, ref_K, scale_factor)
            cost_list.append(cost)  # (b, c,h, w)
        # cost = torch.stack(cost_list, dim=1).min(dim=1)[0]
        cost = torch.stack(cost_list, dim=1).mean(dim=1)
        return cost

    def forward(self, target_image, ref_imgs, intrinsics):
        """ Estimate inv depth and  poses """
        # run the feature network
        fmaps = self.fnet(torch.cat([target_image] + ref_imgs, dim=0))
        fmaps = torch.split(
            fmaps, [target_image.shape[0]] * (1 + len(ref_imgs)), dim=0)
        fmap1, fmaps_ref = fmaps[0], fmaps[1:]
        assert target_image.shape[2] / fmap1.shape[2] == self.feat_ratio

        # initial pose
        pose_list_init = []
        for fmap_ref in fmaps_ref:
            pose_list_init.append(
                self.pose_head(torch.cat([fmap1, fmap_ref], dim=1)))

        # initial depth
        inv_depth_init = self.depth_head(fmap1, act_fn=F.sigmoid)
        up_mask = self.upmask_net(fmap1)
        inv_depth_up_init = self.upsample_depth(
            inv_depth_init, up_mask, ratio=self.feat_ratio)

        inv_depth_predictions = [self.scale_inv_depth(inv_depth_up_init)[0]]
        pose_predictions = [[pose.clone() for pose in pose_list_init]]

        # run the context network for optimization
        if self.iters > 0:
            cnet_depth = self.cnet_depth(target_image)
            hidden_d, inp_d = torch.split(
                cnet_depth, [self.hdim, self.cdim], dim=1)
            hidden_d = torch.tanh(hidden_d)
            inp_d = torch.relu(inp_d)

            img_pairs = []
            for ref_img in ref_imgs:
                img_pairs.append(torch.cat([target_image, ref_img], dim=1))
            cnet_pose_list = self.cnet_pose(img_pairs)
            hidden_p_list, inp_p_list = [], []
            for cnet_pose in cnet_pose_list:
                hidden_p, inp_p = torch.split(
                    cnet_pose, [self.hdim, self.cdim], dim=1)
                hidden_p_list.append(torch.tanh(hidden_p))
                inp_p_list.append(torch.relu(inp_p))

        # optimization start.................
        pose_list = pose_list_init
        inv_depth = inv_depth_init
        inv_depth_up = None
        for itr in range(self.iters):
            inv_depth = inv_depth.detach()
            pose_list = [pose.detach() for pose in pose_list]

            # calc cost
            pose_cost_func_list = []
            for fmap_ref in fmaps_ref:
                pose_cost_func_list.append(
                    partial(
                        self.get_cost_each,
                        fmap=fmap1,
                        fmap_ref=fmap_ref,
                        depth=inv2depth(self.scale_inv_depth(inv_depth)[0]),
                        K=intrinsics,
                        ref_K=intrinsics,
                        scale_factor=1.0 / self.feat_ratio))

            depth_cost_func = partial(
                self.depth_cost_calc,
                fmap=fmap1,
                fmaps_ref=fmaps_ref,
                pose_list=pose_list,
                K=intrinsics,
                ref_K=intrinsics,
                scale_factor=1.0 / self.feat_ratio)

            # ########  update depth ##########
            hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block_depth(
                hidden_d,
                depth_cost_func,
                inv_depth,
                inp_d,
                seq_len=self.seq_len,
                scale_func=self.scale_inv_depth)

            if not self.inter_sup:
                up_mask_seqs, inv_depth_seqs = [up_mask_seqs[-1]
                                                ], [inv_depth_seqs[-1]]
            # upsample predictions
            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                inv_depth_up = self.upsample_depth(
                    inv_depth_i, up_mask_i, ratio=self.feat_ratio)
                inv_depth_predictions.append(
                    self.scale_inv_depth(inv_depth_up)[0])
            inv_depth = inv_depth_seqs[-1]

            # ########  update pose ###########
            pose_list_seqs = [None] * len(pose_list)
            for i, (pose,
                    hidden_p) in enumerate(zip(pose_list, hidden_p_list)):
                hidden_p, pose_seqs = self.update_block_pose(
                    hidden_p,
                    pose_cost_func_list[i],
                    pose,
                    inp_p_list[i],
                    seq_len=self.seq_len)
                hidden_p_list[i] = hidden_p
                if not self.inter_sup:
                    pose_seqs = [pose_seqs[-1]]
                pose_list_seqs[i] = pose_seqs

            for pose_list_i in zip(*pose_list_seqs):
                pose_predictions.append([pose.clone() for pose in pose_list_i])

            pose_list = list(zip(*pose_list_seqs))[-1]

        if not self.training:
            return inv_depth_predictions[-1], \
                torch.stack(pose_predictions[-1], dim=1).view(target_image.shape[0], len(ref_imgs), 6)  # (b, n, 6)

        return inv_depth_predictions, \
            torch.stack([torch.stack(poses_ref, dim=1) for poses_ref in pose_predictions], dim=2)  # (b, n, iters, 6)

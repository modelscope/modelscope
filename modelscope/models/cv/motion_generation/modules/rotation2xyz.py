# This code is borrowed and modified from Human Motion Diffusion Model,
# made publicly available under MIT license at https://github.com/GuyTevet/motion-diffusion-model

import torch

from modelscope.utils.cv.motion_utils import rotation_conversions as geometry
from .smpl import JOINTSTYPE_ROOT, SMPL

JOINTSTYPES = ['a2m', 'a2mpl', 'smpl', 'vibe', 'vertices']


class Rotation2xyz:

    def __init__(self, device, smpl_data_path, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL(smpl_data_path).eval().to(device)

    def __call__(self,
                 x,
                 mask,
                 pose_rep,
                 translation,
                 glob,
                 jointstype,
                 vertstrans,
                 betas=None,
                 beta=0,
                 glob_rot=None,
                 get_rotations_back=False,
                 **kwargs):
        if pose_rep == 'xyz':
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]),
                              dtype=bool,
                              device=x.device)

        if not glob and glob_rot is None:
            raise TypeError(
                'You must specify global rotation if glob is False')

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError('This jointstype is not implemented.')

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == 'rotvec':
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == 'rotmat':
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == 'rotquat':
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == 'rot6d':
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError('No geometry for this one.')

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(
                1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros(
                [rotations.shape[0], self.smpl_model.num_betas],
                dtype=rotations.dtype,
                device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()
        out = self.smpl_model(
            body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(
            nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != 'vertices':
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz

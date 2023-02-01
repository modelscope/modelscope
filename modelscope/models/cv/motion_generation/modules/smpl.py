# This code is borrowed and modified from Human Motion Diffusion Model,
# made publicly available under MIT license at https://github.com/GuyTevet/motion-diffusion-model

import contextlib
import os.path as osp

import numpy as np
import torch
from smplx import SMPLLayer as _SMPLLayer
from smplx.lbs import vertices2joints

action2motion_joints = [
    8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38
]

JOINTSTYPE_ROOT = {
    'a2m': 0,  # action2motion
    'smpl': 0,
    'a2mpl': 0,  # set(smpl, a2m)
    'vibe': 8
}  # 0 is the 8 position: OP MidHip below

JOINT_MAP = {
    'OP Nose': 24,
    'OP Neck': 12,
    'OP RShoulder': 17,
    'OP RElbow': 19,
    'OP RWrist': 21,
    'OP LShoulder': 16,
    'OP LElbow': 18,
    'OP LWrist': 20,
    'OP MidHip': 0,
    'OP RHip': 2,
    'OP RKnee': 5,
    'OP RAnkle': 8,
    'OP LHip': 1,
    'OP LKnee': 4,
    'OP LAnkle': 7,
    'OP REye': 25,
    'OP LEye': 26,
    'OP REar': 27,
    'OP LEar': 28,
    'OP LBigToe': 29,
    'OP LSmallToe': 30,
    'OP LHeel': 31,
    'OP RBigToe': 32,
    'OP RSmallToe': 33,
    'OP RHeel': 34,
    'Right Ankle': 8,
    'Right Knee': 5,
    'Right Hip': 45,
    'Left Hip': 46,
    'Left Knee': 4,
    'Left Ankle': 7,
    'Right Wrist': 21,
    'Right Elbow': 19,
    'Right Shoulder': 17,
    'Left Shoulder': 16,
    'Left Elbow': 18,
    'Left Wrist': 20,
    'Neck (LSP)': 47,
    'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49,
    'Thorax (MPII)': 50,
    'Spine (H36M)': 51,
    'Jaw (H36M)': 52,
    'Head (H36M)': 53,
    'Nose': 24,
    'Left Eye': 26,
    'Right Eye': 25,
    'Left Ear': 28,
    'Right Ear': 27
}

JOINT_NAMES = list(JOINT_MAP.keys())


class SMPL(_SMPLLayer):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, smpl_data_path, **kwargs):
        kwargs['model_path'] = osp.join(smpl_data_path, 'SMPL_NEUTRAL.pkl')

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)

        J_regressor_extra = np.load(
            osp.join(smpl_data_path, 'J_regressor_extra.npy'))
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        vibe_indexes = np.array([JOINT_MAP[i] for i in JOINT_NAMES])
        a2m_indexes = vibe_indexes[action2motion_joints]
        smpl_indexes = np.arange(24)
        a2mpl_indexes = np.unique(np.r_[smpl_indexes, a2m_indexes])

        self.maps = {
            'vibe': vibe_indexes,
            'a2m': a2m_indexes,
            'smpl': smpl_indexes,
            'a2mpl': a2mpl_indexes
        }

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        extra_joints = vertices2joints(self.J_regressor_extra,
                                       smpl_output.vertices)
        all_joints = torch.cat([smpl_output.joints, extra_joints], dim=1)

        output = {'vertices': smpl_output.vertices}

        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]

        return output

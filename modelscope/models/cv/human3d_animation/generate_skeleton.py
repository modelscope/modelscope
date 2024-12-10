# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle

import numpy as np
import torch

from .bvh_writer import WriterWrapper
from .utils import matrix_to_axis_angle, rotation_6d_to_matrix


def load_smpl_params(pose_fname):
    with open(pose_fname, 'rb') as f:
        data = pickle.load(f)
        pose = torch.from_numpy(data['pose'])
        beta = torch.from_numpy(data['betas'])
        trans = torch.from_numpy(data['trans'])
        if 'joints' in data:
            joints = torch.from_numpy(data['joints'])
            joints = joints.reshape(1, -1, 3)
        else:
            joints = None
    trans = trans.reshape(1, 3)
    beta = beta.reshape(1, -1)[:, :10]
    pose = pose.reshape(-1, 24 * 3)
    return pose, beta, trans, joints


def set_pose_param(pose, start, end):
    pose[:, start * 3:(end + 1) * 3] = 0
    return pose


def load_test_anim(filename, device, mode='move'):
    anim = np.load(filename)
    anim = torch.tensor(anim, device=device, dtype=torch.float)
    poses = anim[:, :-3]
    loc = anim[:, -3:]
    if os.path.basename(filename)[:5] == 'comb_':
        loc = loc / 100
    repeat = 0
    idx = -1
    for i in range(poses.shape[0]):
        if i == 0:
            continue
        if repeat >= 5:
            idx = i
            break
        if poses[i].equal(poses[i - 1]):
            repeat += 1
        else:
            repeat = 0
    poses = poses[:idx - 5, :]
    loc = loc[:idx - 5, :]

    if mode == 'inplace':
        loc[1:, :] = loc[0, :]

    return poses, loc


def load_syn_motion(filename, device, mode='move'):
    data = np.load(filename, allow_pickle=True).item()
    anim = data['thetas']
    n_joint, c, t = anim.shape

    anim = torch.tensor(anim, device=device, dtype=torch.float)
    anim = anim.permute(2, 0, 1)  # 180, 24, 6
    poses = anim.reshape(-1, 6)
    poses = rotation_6d_to_matrix(poses)
    poses = matrix_to_axis_angle(poses)
    poses = poses.reshape(-1, 24, 3)

    loc = data['root_translation']
    loc = torch.tensor(loc, device=device, dtype=torch.float)
    loc = loc.permute(1, 0)

    if mode == 'inplace':
        loc = torch.zeros((t, 3))

    print('load %s' % filename)

    return poses, loc


def load_action(action_name,
                model_dir,
                action_dir,
                mode='move',
                device=torch.device('cpu')):
    action_path = os.path.join(action_dir, action_name + '.npy')
    if not os.path.exists(action_path):
        print('can not find action %s, use default action instead' %
              (action_name))
        action_path = os.path.join(model_dir, '3D-assets', 'SwingDancing.npy')
    print('load action %s' % action_path)
    test_pose, test_loc = load_test_anim(
        action_path, device, mode=mode)  # pose:[45,72], loc:[45,1,3]

    return test_pose, test_loc


def load_action_list(action,
                     model_dir,
                     action_dir,
                     mode='move',
                     device=torch.device('cpu')):
    action_list = action.split(',')
    test_pose, test_loc = load_action(
        action_list[0], model_dir, action_dir, mode=mode, device=device)
    final_loc = test_loc[-1, :]
    idx = 0
    if len(action_list) > 1:
        for action in action_list:
            if idx == 0:
                idx += 1
                continue
            print('load action %s' % action)
            pose, loc = load_action(
                action, model_dir, action_dir, mode=mode, device=device)
            delta_loc = final_loc - loc[0, :]
            loc += delta_loc
            final_loc = loc[-1, :]
            test_pose = torch.cat([test_pose, pose], 0)
            test_loc = torch.cat([test_loc, loc], 0)
        idx += 1
    return test_pose, test_loc


def gen_skeleton_bvh(model_dir, action_dir, case_dir, action, mode='move'):
    outpath_a = os.path.join(case_dir, 'skeleton_a.bvh')
    device = torch.device('cpu')
    assets_dir = os.path.join(model_dir, '3D-assets')
    pkl_path = os.path.join(assets_dir, 'smpl.pkl')
    poses, shapes, trans, joints = load_smpl_params(pkl_path)
    if action.endswith('.npy'):
        skeleton_path = os.path.join(assets_dir, 'skeleton_nohand.npy')
    else:
        skeleton_path = os.path.join(assets_dir, 'skeleton.npy')
    data = np.load(skeleton_path, allow_pickle=True).item()
    skeleton = data['skeleton']
    parent = data['parent']
    skeleton = skeleton.squeeze(0)
    bvh_writer = WriterWrapper(parent)

    if action.endswith('.npy'):
        action_path = action
        print('load action %s' % action_path)
        test_pose, test_loc = load_syn_motion(action_path, device, mode=mode)
        bvh_writer.write(
            outpath_a,
            skeleton,
            test_pose,
            action_loc=test_loc,
            rest_pose=poses)

    else:
        print('load action %s' % action)
        test_pose, test_loc = load_action_list(
            action, model_dir, action_dir, mode='move', device=device)
        std_y = torch.tensor(0.99)
        test_loc = test_loc + (skeleton[0, 1] - std_y)
        bvh_writer.write(outpath_a, skeleton, test_pose, action_loc=test_loc)

    print('save %s' % outpath_a)

    return 0

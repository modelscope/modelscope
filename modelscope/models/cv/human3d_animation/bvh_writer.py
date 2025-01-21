# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch

from .transforms import aa2quat, batch_rodrigues, mat2aa, quat2euler


def write_bvh(parent,
              offset,
              rotation,
              position,
              names,
              frametime,
              order,
              path,
              endsite=None):
    file = open(path, 'w')
    frame = rotation.shape[0]
    joint_num = rotation.shape[1]
    order = order.upper()

    file_string = 'HIERARCHY\n'

    seq = []

    def write_static(idx, prefix):
        nonlocal parent, offset, rotation, names
        nonlocal order, endsite, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition \
            {}rotation {}rotation {}rotation'.format(*order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation \
            {}rotation'.format(*order)
        offset_label = 'OFFSET %.6f %.6f %.6f' % (
            offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx + 1, rotation.shape[1]):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(
        frame) + 'Frame Time: %.8f\n' % frametime
    for i in range(frame):
        file_string += '%.6f %.6f %.6f ' % (position[i][0], position[i][1],
                                            position[i][2])

        for j in range(joint_num):
            idx = seq[j]
            file_string += '%.6f %.6f %.6f ' % (
                rotation[i][idx][0], rotation[i][idx][1], rotation[i][idx][2])

        file_string += '\n'

    file.write(file_string)
    return file_string


class WriterWrapper:

    def __init__(self, parents):
        self.parents = parents

    def axis2euler(self, rot):
        rot = rot.reshape(rot.shape[0], -1, 3)  # 45, 24, 3
        quat = aa2quat(rot)
        euler = quat2euler(quat, order='xyz')
        rot = euler
        return rot

    def mapper_rot_mixamo(self, rot, n_bone):
        rot = rot.reshape(rot.shape[0], -1, 3)

        smpl_mapper = [
            0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 17, 21, 15, 18, 22, 19,
            23, 20, 24
        ]

        if n_bone > 24:
            hand_mapper = list(range(25, 65))
            smpl_mapper += hand_mapper

        new_rot = torch.zeros((rot.shape[0], n_bone, 3))  # n, 24, 3
        new_rot[:, :len(smpl_mapper), :] = rot[:, smpl_mapper, :]

        return new_rot

    def transform_rot_with_restpose(self, rot, rest_pose, node_list, n_bone):

        rest_pose = batch_rodrigues(rest_pose.reshape(-1, 3)).reshape(
            1, n_bone, 3, 3)  # N*3-> N*3*3

        frame_num = rot.shape[0]
        rot = rot.reshape(rot.shape[0], -1, 3)
        new_rot = rot.clone()
        for k in range(frame_num):
            action_rot = batch_rodrigues(rot[k].reshape(-1, 3)).reshape(
                1, n_bone, 3, 3)
            for i in node_list:
                rot1 = rest_pose[0, i, :, :]
                rot2 = action_rot[0, i, :, :]
                nrot = torch.matmul(rot2, torch.inverse(rot1))
                nvec = mat2aa(nrot)
                new_rot[k, i, :] = nvec

        new_rot = self.axis2euler(new_rot)  # =# 45,24,3
        return new_rot

    def transform_rot_with_stdApose(self, rot, rest_pose):
        print('transform_rot_with_stdApose')
        rot = rot.reshape(rot.shape[0], -1, 3)
        rest_pose = self.axis2euler(rest_pose)
        print(rot.shape)
        print(rest_pose.shape)
        smpl_left_arm_idx = 18
        smpl_right_arm_idx = 19
        std_arm_rot = torch.tensor([[21.7184, -4.8148, 16.3985],
                                    [-20.1108, 10.7190, -8.9279]])
        x = rest_pose[:, smpl_left_arm_idx:smpl_right_arm_idx + 1, :]
        delta = (x - std_arm_rot)
        rot[:, smpl_left_arm_idx:smpl_right_arm_idx + 1, :] -= delta
        return rot

    def write(self,
              filename,
              offset,
              rot=None,
              action_loc=None,
              rest_pose=None,
              correct_arm=0):  # offset: [24,3], rot:[45,72]
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset)
        n_bone = offset.shape[0]  # 24
        pos = offset[0].unsqueeze(0)  # 1,3

        if rot is None:
            rot = np.zeros((1, n_bone, 3))
        else:  # rot: 45, 72
            if rest_pose is None:
                rot = self.mapper_rot_mixamo(rot, n_bone)
            else:
                if correct_arm == 1:
                    rot = self.mapper_rot_mixamo(rot, n_bone)
                    print(rot.shape)
                    node_list_chage = [16, 17]
                    n_bone = rot.shape[1]
                    print(rot[0, 19, :])
                else:
                    node_list_chage = [1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]
                    rot = self.transform_rot_with_restpose(
                        rot, rest_pose, node_list_chage, n_bone)

            rest = torch.zeros((1, n_bone * 3))
            rest = self.axis2euler(rest)
            frames_add = 1
            rest = rest.repeat(frames_add, 1, 1)
            rot = torch.cat((rest, rot), 0)

        pos = pos.repeat(rot.shape[0], 1)
        action_len = action_loc.shape[0]
        pos[-action_len:, :] = action_loc[..., :]

        names = ['%02d' % i for i in range(n_bone)]
        write_bvh(self.parents, offset, rot, pos, names, 0.0333, 'xyz',
                  filename)

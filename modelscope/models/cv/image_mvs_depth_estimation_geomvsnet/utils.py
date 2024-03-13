# The implementation here is modified based on https://github.com/xy-guo/MVSNet_pytorch
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):

    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError(
            'invalid input type {} for tensor2numpy'.format(type(vars)))


@make_recursive_func
def numpy2torch(vars):
    if isinstance(vars, np.ndarray):
        return torch.from_numpy(vars)
    elif isinstance(vars, torch.Tensor):
        return vars
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for numpy2torch'.format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device('cuda'))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for tensor2numpy'.format(type(vars)))


def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]  # rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append('%f %f %f %d %d %d 0\n' %
                          (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, 'w')
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), ''.join(points)))
    file.close()


def write_cam(file, cam):
    f = open(file, 'w')
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' '
            + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

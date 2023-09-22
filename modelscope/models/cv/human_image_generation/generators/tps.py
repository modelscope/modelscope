import torch
import torch.nn.functional as F
from torch import nn


class TPS(nn.Module):

    def __init__(self, mode='kp'):
        super().__init__()
        self.mode = mode

    def trans(self, kp_1):
        if self.mode == 'kp':
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(
                kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
            K = K**2
            K = K * torch.log(K + 1e-9)

            one1 = torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2],
                              1).to(device).type(kp_type)
            kp_1p = torch.cat([kp_1, one1], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3,
                               3).to(device).type(kp_type)
            P = torch.cat([kp_1p, zero], 2)
            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            L = torch.cat([L, P], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3,
                               2).to(device).type(kp_type)
            kp_substitute = torch.zeros(kp_1.shape).to(device).type(kp_type)
            Y = torch.cat([kp_substitute, zero], 2)
            one = torch.eye(L.shape[2]).expand(
                L.shape).to(device).type(kp_type) * 0.01
            L = L + one

            param = torch.matmul(torch.inverse(L), Y)
            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)

            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]
        else:
            raise Exception('Error TPS mode')

    def transform_frame(self, frame):
        grid = make_coordinate_grid(
            frame.shape[2:], type=frame.type()).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == 'kp':
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(
            coordinates.device)
        control_params = self.control_params.type(coordinates.type()).to(
            coordinates.device)

        if self.mode == 'kp':
            transformed = torch.matmul(theta[:, :, :, :2],
                                       coordinates.permute(
                                           0, 2, 1)) + theta[:, :, :, 2:]

            distances = coordinates.view(
                coordinates.shape[0], 1, 1, -1, 2) - control_points.view(
                    self.bs, control_points.shape[1], -1, 1, 2)
            distances = distances**2
            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result

        else:
            raise Exception('Error TPS mode')

        return transformed

    def preprocess_kp(self, kp_1):
        '''
            kp_1: (b, ntps*nkp, 2)
        '''
        kp_mask = (kp_1 == -1)
        num_keypoints = kp_1.shape[1]
        kp_1 = kp_1.masked_fill(kp_mask, -1.)
        return kp_1, num_keypoints

    def forward(self, source_image, kp_driving):
        bs, _, h, w = source_image.shape
        self.bs = bs
        kp_driving, num_keypoints = self.preprocess_kp(kp_driving)
        kp_1 = kp_driving.view(bs, -1, num_keypoints, 2)
        self.trans(kp_1)
        grid = self.transform_frame(source_image)
        grid = grid.view(bs, h, w, 2)
        return grid


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

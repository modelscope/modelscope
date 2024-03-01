import torch
import torch.nn.functional as F

from modelscope.utils.logger import get_logger

logger = get_logger()


class Intrinsics:
    """Intrinsics"""

    def __init__(self, width, height, fu, fv, cu=0, cv=0):
        self.height, self.width = height, width
        self.fu, self.fv = fu, fv  # fu, fv: focal length along the horizontal and vertical axes

        # cu, cv: optical center along the horizontal and vertical axes
        self.cu = cu if cu > 0 else (width - 1) / 2.0
        self.cv = cv if cv > 0 else (height - 1) / 2.0

        # U, V represent the homogeneous horizontal and vertical coordinates in the pixel space
        self.U = torch.arange(start=0, end=width).expand(height, width).float()
        self.V = torch.arange(
            start=0, end=height).expand(width, height).t().float()

        # X_cam, Y_cam represent the homogeneous x, y coordinates (assuming depth z=1) in the camera coordinate system
        self.X_cam = (self.U - self.cu) / self.fu
        self.Y_cam = (self.V - self.cv) / self.fv

        self.is_cuda = False

    def cuda(self):
        self.X_cam.data = self.X_cam.data.cuda()
        self.Y_cam.data = self.Y_cam.data.cuda()
        self.is_cuda = True
        return self

    def scale(self, height, width):
        # return a new set of corresponding intrinsic parameters for the scaled image
        ratio_u = float(width) / self.width
        ratio_v = float(height) / self.height
        fu = ratio_u * self.fu
        fv = ratio_v * self.fv
        cu = ratio_u * self.cu
        cv = ratio_v * self.cv
        new_intrinsics = Intrinsics(width, height, fu, fv, cu, cv)
        if self.is_cuda:
            new_intrinsics.cuda()
        return new_intrinsics

    def __print__(self):
        logger.info(
            'size=({},{})\nfocal length=({},{})\noptical center=({},{})'.
            format(self.height, self.width, self.fv, self.fu, self.cv,
                   self.cu))


def image_to_pointcloud(depth, intrinsics):
    assert depth.dim() == 4
    assert depth.size(1) == 1

    X = depth * intrinsics.X_cam
    Y = depth * intrinsics.Y_cam
    return torch.cat((X, Y, depth), dim=1)


def pointcloud_to_image(pointcloud, intrinsics):
    assert pointcloud.dim() == 4

    batch_size = pointcloud.size(0)
    X = pointcloud[:, 0, :, :]  # .view(batch_size, -1)
    Y = pointcloud[:, 1, :, :]  # .view(batch_size, -1)
    Z = pointcloud[:, 2, :, :].clamp(min=1e-3)  # .view(batch_size, -1)

    # compute pixel coordinates
    U_proj = intrinsics.fu * X / Z + intrinsics.cu  # horizontal pixel coordinate
    V_proj = intrinsics.fv * Y / Z + intrinsics.cv  # vertical pixel coordinate

    # normalization to [-1, 1], required by torch.nn.functional.grid_sample
    w = intrinsics.width
    h = intrinsics.height
    U_proj_normalized = (2 * U_proj / (w - 1) - 1).view(batch_size, -1)
    V_proj_normalized = (2 * V_proj / (h - 1) - 1).view(batch_size, -1)

    # This was important since PyTorch didn't do as it claimed for points out of boundary
    # See https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    # Might not be necessary any more
    U_proj_mask = ((U_proj_normalized > 1) + (U_proj_normalized < -1)).detach()
    U_proj_normalized[U_proj_mask] = 2
    V_proj_mask = ((V_proj_normalized > 1) + (V_proj_normalized < -1)).detach()
    V_proj_normalized[V_proj_mask] = 2

    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized],
                               dim=2)  # [B, H*W, 2]
    return pixel_coords.view(batch_size, intrinsics.height, intrinsics.width,
                             2)


def batch_multiply(batch_scalar, batch_matrix):
    # input: batch_scalar of size b, batch_matrix of size b * 3 * 3
    # output: batch_matrix of size b * 3 * 3
    batch_size = batch_scalar.size(0)
    output = batch_matrix.clone()
    for i in range(batch_size):
        output[i] = batch_scalar[i] * batch_matrix[i]
    return output


def transform_curr_to_near(pointcloud_curr, r_mat, t_vec, intrinsics):
    # translation and rotmat represent the transformation from tgt pose to src pose
    batch_size = pointcloud_curr.size(0)
    XYZ_ = torch.bmm(r_mat, pointcloud_curr.view(batch_size, 3, -1))

    X = (XYZ_[:, 0, :] + t_vec[:, 0].unsqueeze(1)).view(
        -1, 1, intrinsics.height, intrinsics.width)
    Y = (XYZ_[:, 1, :] + t_vec[:, 1].unsqueeze(1)).view(
        -1, 1, intrinsics.height, intrinsics.width)
    Z = (XYZ_[:, 2, :] + t_vec[:, 2].unsqueeze(1)).view(
        -1, 1, intrinsics.height, intrinsics.width)

    pointcloud_near = torch.cat((X, Y, Z), dim=1)

    return pointcloud_near


def homography_from(rgb_near, depth_curr, r_mat, t_vec, intrinsics):
    # inverse warp the RGB image from the nearby frame to the current frame

    # to ensure dimension consistency
    r_mat = r_mat.view(-1, 3, 3)
    t_vec = t_vec.view(-1, 3)

    # compute source pixel coordinate
    pointcloud_curr = image_to_pointcloud(depth_curr, intrinsics)
    pointcloud_near = transform_curr_to_near(pointcloud_curr, r_mat, t_vec,
                                             intrinsics)
    pixel_coords_near = pointcloud_to_image(pointcloud_near, intrinsics)

    # the warping
    warped = F.grid_sample(rgb_near, pixel_coords_near)

    return warped

# @Time    : 2018-9-21 14:36
# @Author  : xylon
# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import numpy as np
import torch


def distance_matrix_vector(anchor, positive):
    """
    Given batch of anchor descriptors and positive descriptors calculate distance matrix
    :param anchor: (B, 128)
    :param positive: (B, 128)
    :return:
    """
    eps = 1e-8
    FeatSimi_Mat = 2 - 2 * torch.mm(anchor, positive.t())  # [0, 4]
    FeatSimi_Mat = FeatSimi_Mat.clamp(min=eps, max=4.0)
    FeatSimi_Mat = torch.sqrt(FeatSimi_Mat)  # euc [0, 2]
    return FeatSimi_Mat


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = y.transpose(0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = x.transpose(0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    eps = 1e-8
    return torch.sqrt(dist.clamp(min=eps, max=np.inf))


def ptCltoCr(leftC, homolr, right_imscale, right_imorint=None, clamp=True):
    """
    ptCltoCr is the abbreviation of projective transform keypoints Coordinates in left back to Coordinates in right
    :param leftC: tensor #(B*topk, 4): the 4 correspond to (b, y, x, 0) each element in it has length B*topk
    :param homolr: torch(B, 3, 3): homogeneous matrix
    :param right_imscale: (B, H, W, 1)
    :param right_imorint: (B, H, W, 1, 2)
    :param clamp: whether clamp rightC_homo
    :return: tuple (b, y, x, 0) each element in that has length B*topk
    """
    # projective transform im1_C back to im2 called im2_Cw
    B, maxh, maxw, C = right_imscale.size(
    )  # tuple (b, h, w) max size of image
    leftC_homo = leftC.clone()
    leftC_homo[:, 3] = leftC_homo[:, 3] + 1  # (B*topk, 4) (b, y, x, 1)
    leftC_homo = leftC_homo[:, 1:]  # (B*topk, 3) (y, x, 1)
    leftC_homo = leftC_homo.index_select(1, leftC_homo.new_tensor(
        [1, 0, 2]))  # (B*topk, 3) [[x], [y], [1]]
    leftC_homo = leftC_homo.view(B, -1, 3)  # (B, topk, 3)
    leftC_homo = leftC_homo.permute(0, 2, 1)  # (B, 3, topk)

    rightC_homo = torch.matmul(homolr,
                               leftC_homo.float())  # (B, 3, topk) (x, y, h)
    rightC_homo = rightC_homo.permute(0, 2, 1)  # (B, topk, 3) (x, y, h)
    # (B, topk, 3) (x, y, h) to 1
    rightC_homo = rightC_homo / (
        torch.unsqueeze(rightC_homo[:, :, 2], -1) + 1e-8)
    rightC_homo = rightC_homo.round().long()
    if clamp:
        rightC_homo[:, :, 0] = rightC_homo[:, :, 0].clamp(min=0, max=maxw - 1)
        rightC_homo[:, :, 1] = rightC_homo[:, :, 1].clamp(min=0, max=maxh - 1)

    topk = rightC_homo.size(1)
    batch_v = (torch.arange(B, device=rightC_homo.device).view(B, 1, 1).repeat(
        1, topk, 1))  # (B, topk, 1)
    # (B, topk, 4) (B, x, y, h)
    rightC_homo = torch.cat((batch_v, rightC_homo), -1)
    rightC_homo = rightC_homo.contiguous().view(-1,
                                                4)  # (B*topk, 4) (B, x, y, h)
    rightC_homo = rightC_homo.index_select(
        1, rightC_homo.new_tensor([0, 2, 1, 3]))  # (B*topk, 4) (B, y, x, h)
    rightC_homo[:, 3] = rightC_homo[:, 3] - 1  # (B*topk, 4) (B, y, x, 0)

    right_imS = right_imscale.view(-1)  # (B*H*W)
    dim1 = maxw
    dim2 = maxh * maxw
    scale_idx = rightC_homo[:,
                            0] * dim2 + rightC_homo[:,
                                                    1] * dim1 + rightC_homo[:,
                                                                            2]
    scale_idx = scale_idx.clamp(min=0, max=dim2 * B - 1)
    right_imS = right_imS.gather(0, scale_idx)  # (B*topk)

    if right_imorint is None:
        right_imO = None
    else:
        right_cos, right_sin = right_imorint.squeeze().chunk(
            chunks=2, dim=-1)  # each is (B, H, W, 1)
        right_cos = right_cos.view(-1)  # (B*H*W)
        right_sin = right_sin.view(-1)  # (B*H*W)
        right_cos = right_cos.gather(0, scale_idx)  # (B*topk)
        right_sin = right_sin.gather(0, scale_idx)  # (B*topk)
        right_imO = torch.cat(
            (right_cos.unsqueeze(-1), right_sin.unsqueeze(-1)),
            dim=-1)  # (B*topk, 2)

    return rightC_homo, right_imS, right_imO


def L2Norm(input, dim=-1):
    input = input / torch.norm(input, p=2, dim=dim, keepdim=True)
    return input


def MSD(x, y):
    """
    mean square distance
    :param x: (B, H, W, 2) 2 corresponds to XY
    :param y: (B, H, W, 2) 2 corresponds to XY
    :return: distance: (B, H, W, 1)
    """
    sub = x - y
    square = sub**2
    sm = square.sum(keepdim=True, dim=-1)
    sqr = torch.sqrt((sm + 1e-8).float())
    return sqr * 2

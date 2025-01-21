# The implementation is adopted from https://github.com/TreB1eN/InsightFace_Pytorch/tree/master/mtcnn_pytorch

import cv2
import numpy as np

from .matlab_cp2tform import get_similarity_transform_for_cv2

# reference facial points, a list of coordinates (x,y)
dx = 1
dy = 1
REFERENCE_FACIAL_POINTS = [
    [30.29459953 + dx, 51.69630051 + dy],  # left eye
    [65.53179932 + dx, 51.50139999 + dy],  # right eye
    [48.02519989 + dx, 71.73660278 + dy],  # nose
    [33.54930115 + dx, 92.3655014 + dy],  # left mouth
    [62.72990036 + dx, 92.20410156 + dy]  # right mouth
]

DEFAULT_CROP_SIZE = (96, 112)

global FACIAL_POINTS


class FaceWarpException(Exception):

    def __str__(self):
        return 'In File {}:{}'.format(__file__, super.__str__(self))


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    h_crop = tmp_crop_size[0]
    w_crop = tmp_crop_size[1]
    if (output_size):
        if (output_size[0] == h_crop and output_size[1] == w_crop):
            return tmp_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(
                    tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    factor = inner_padding_factor > 0 or outer_padding[0] > 0
    factor = factor or outer_padding[1] > 0
    if (factor and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    cond1 = outer_padding[0] < output_size[0]
    cond2 = outer_padding[1] < output_size[1]
    if not (cond1 and cond2):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) pad the inner region according inner_padding_factor
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # 2) resize the padded inner region
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[
            1] * tmp_crop_size[0]:
        raise FaceWarpException(
            'Must have (output_size - outer_padding)'
            '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):

    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([[A[0, 0], A[1, 0], A[2, 0]],
                          [A[0, 1], A[1, 1], A[2, 1]]])
    elif rank == 2:
        tfm = np.float32([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]])

    return tfm


def warp_and_crop_face(src_img,
                       facial_pts,
                       ratio=0.84,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='similarity'
                       '',
                       return_trans_inv=False):

    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding,
                default_square)

    ref_pts = np.float32(reference_pts)

    factor = ratio
    ref_pts = (ref_pts - 112 / 2) * factor + 112 / 2
    ref_pts *= crop_size[0] / 112.

    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts, ref_pts)
        tfm_inv = cv2.getAffineTransform(ref_pts, src_pts)

    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
        tfm_inv = get_affine_transform_matrix(ref_pts, src_pts)
    else:
        tfm, tfm_inv = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(
        src_img,
        tfm, (crop_size[0], crop_size[1]),
        borderValue=(255, 255, 255))

    if return_trans_inv:
        return face_img, tfm_inv
    else:
        return face_img

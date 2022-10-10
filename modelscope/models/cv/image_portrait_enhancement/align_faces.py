# Part of the implementation is borrowed and modified from Face-Alignment,
# publicly available at https://github.com/foamliu/Face-Alignment/blob/master/align_faces.py
import cv2
import numpy as np
from skimage import transform as trans

from modelscope.utils.logger import get_logger

logger = get_logger()

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [[30.29459953, 51.69630051],
                           [65.53179932, 51.50139999],
                           [48.02519989,
                            71.73660278], [33.54930115, 92.3655014],
                           [62.72990036, 92.20410156]]

DEFAULT_CROP_SIZE = (96, 112)


def _umeyama(src, dst, estimate_scale=True, scale=1.0):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim, ), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = scale

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T, scale


class FaceWarpException(Exception):

    def __str__(self):
        return 'In File {}:{}'.format(__file__, super.__str__(self))


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    ref_5pts = np.array(REFERENCE_FACIAL_POINTS)
    ref_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(ref_crop_size) - ref_crop_size
        ref_5pts += size_diff / 2
        ref_crop_size += size_diff

    if (output_size and output_size[0] == ref_crop_size[0]
            and output_size[1] == ref_crop_size[1]):
        return ref_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            logger.info('No paddings to do: return default reference points')
            return ref_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(
                    ref_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0
         or outer_padding[1] > 0) and output_size is None):
        output_size = ref_crop_size * (1 + inner_padding_factor * 2).astype(
            np.int32)
        output_size += np.array(outer_padding)
        logger.info('deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) pad the inner region according inner_padding_factor
    if inner_padding_factor > 0:
        size_diff = ref_crop_size * inner_padding_factor * 2
        ref_5pts += size_diff / 2
        ref_crop_size += np.round(size_diff).astype(np.int32)

    # 2) resize the padded inner region
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * ref_crop_size[1] != size_bf_outer_pad[
            1] * ref_crop_size[0]:
        raise FaceWarpException(
            'Must have (output_size - outer_padding)'
            '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / ref_crop_size[0]
    ref_5pts = ref_5pts * scale_factor
    ref_crop_size = size_bf_outer_pad

    # 3) add outer_padding to make output_size
    reference_5point = ref_5pts + np.array(outer_padding)
    ref_crop_size = output_size

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


def get_params(reference_pts, facial_pts, align_type):
    ref_pts = np.float32(reference_pts)
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
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
        tfm_inv = cv2.getAffineTransform(ref_pts[0:3], src_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
        tfm_inv = get_affine_transform_matrix(ref_pts, src_pts)
    else:
        params, scale = _umeyama(src_pts, ref_pts)
        tfm = params[:2, :]

        params, _ = _umeyama(ref_pts, src_pts, False, scale=1.0 / scale)
        tfm_inv = params[:2, :]

    return tfm, tfm_inv


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity'):  # smilarity cv2_affine affine

    reference_pts_112 = get_reference_facial_points((112, 112), 0.25, (0, 0),
                                                    True)
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = True  # False
            inner_padding_factor = 0.25  # 0
            outer_padding = (0, 0)
            output_size = crop_size
            reference_pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding,
                default_square)

    tfm, tfm_inv = get_params(reference_pts, facial_pts, align_type)
    tfm_112, tfm_inv_112 = get_params(reference_pts_112, facial_pts,
                                      align_type)

    if src_img is not None:
        face_img = cv2.warpAffine(
            src_img, tfm, (crop_size[0], crop_size[1]), flags=3)
        face_img_112 = cv2.warpAffine(src_img, tfm_112, (112, 112), flags=3)

        return face_img, face_img_112, tfm_inv
    else:
        return tfm, tfm_inv

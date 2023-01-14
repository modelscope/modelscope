# The implementation here is modified based on InsightFace_Pytorch, originally MIT License and publicly available
# at https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/mtcnn_pytorch/src/align_trans.py
import cv2
import numpy as np

from .matlab_cp2tform import get_similarity_transform_for_cv2

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [[30.29459953, 51.69630051],
                           [65.53179932, 51.50139999],
                           [48.02519989,
                            71.73660278], [33.54930115, 92.3655014],
                           [62.72990036, 92.20410156]]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):

    def __str__(self):
        return 'In File {}:{}'.format(__file__, super.__str__(self))


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square:
                crop_size = (112, 112)
            else:
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding)
                = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size and output_size[0] == tmp_crop_size[0]
            and output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(
                    tmp_crop_size))

    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0
         or outer_padding[1] > 0) and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
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
    tmp_crop_size = size_bf_outer_pad

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
    """

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
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity',
                       return_trans_inv=False):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

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
    ref_pts = (ref_pts - 112 / 2) * 0.85 + 112 / 2
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
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm, tfm_inv = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    if return_trans_inv:
        return face_img, tfm_inv
    else:
        return face_img


def get_f5p(landmarks, np_img):
    eye_left = find_pupil(landmarks[36:41], np_img)
    eye_right = find_pupil(landmarks[42:47], np_img)
    if eye_left is None or eye_right is None:
        print('cannot find 5 points with find_pupil, used mean instead.!')
        eye_left = landmarks[36:41].mean(axis=0)
        eye_right = landmarks[42:47].mean(axis=0)
    nose = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    f5p = [[eye_left[0], eye_left[1]], [eye_right[0], eye_right[1]],
           [nose[0], nose[1]], [mouth_left[0], mouth_left[1]],
           [mouth_right[0], mouth_right[1]]]
    return f5p


def find_pupil(landmarks, np_img):
    h, w, _ = np_img.shape
    xmax = int(landmarks[:, 0].max())
    xmin = int(landmarks[:, 0].min())
    ymax = int(landmarks[:, 1].max())
    ymin = int(landmarks[:, 1].min())

    if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > h or xmax > w:
        return None
    eye_img_bgr = np_img[ymin:ymax, xmin:xmax, :]
    eye_img = cv2.cvtColor(eye_img_bgr, cv2.COLOR_BGR2GRAY)
    eye_img = cv2.equalizeHist(eye_img)
    n_marks = landmarks - np.array([xmin, ymin]).reshape([1, 2])
    eye_mask = cv2.fillConvexPoly(
        np.zeros_like(eye_img), n_marks.astype(np.int32), 1)
    ret, thresh = cv2.threshold(eye_img, 100, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = (1 - thresh / 255.) * eye_mask
    cnt = 0
    xm = []
    ym = []
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i, j] > 0.5:
                xm.append(j)
                ym.append(i)
                cnt += 1
    if cnt != 0:
        xm.sort()
        ym.sort()
        xm = xm[cnt // 2]
        ym = ym[cnt // 2]
    else:
        xm = thresh.shape[1] / 2
        ym = thresh.shape[0] / 2

    return xm + xmin, ym + ymin

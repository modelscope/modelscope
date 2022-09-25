"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
"""
import cv2
import numpy as np
from skimage import transform as trans


def align_face(image, size, lmks):
    dst_w = size[1]
    dst_h = size[0]
    # landmark calculation of dst images
    base_w = 96
    base_h = 112
    assert (dst_w >= base_w)
    assert (dst_h >= base_h)
    base_lmk = [
        30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655,
        62.7299, 92.2041
    ]

    dst_lmk = np.array(base_lmk).reshape((5, 2)).astype(np.float32)
    if dst_w != base_w:
        slide = (dst_w - base_w) / 2
        dst_lmk[:, 0] += slide

    if dst_h != base_h:
        slide = (dst_h - base_h) / 2
        dst_lmk[:, 1] += slide

    src_lmk = lmks
    # using skimage method
    tform = trans.SimilarityTransform()
    tform.estimate(src_lmk, dst_lmk)
    t = tform.params[0:2, :]

    assert (image.shape[2] == 3)

    dst_image = cv2.warpAffine(image.copy(), t, (dst_w, dst_h))
    dst_pts = GetAffinePoints(src_lmk, t)
    return dst_image, dst_pts


def GetAffinePoints(pts_in, trans):
    pts_out = pts_in.copy()
    assert (pts_in.shape[1] == 2)

    for k in range(pts_in.shape[0]):
        pts_out[k, 0] = pts_in[k, 0] * trans[0, 0] + pts_in[k, 1] * trans[
            0, 1] + trans[0, 2]
        pts_out[k, 1] = pts_in[k, 0] * trans[1, 0] + pts_in[k, 1] * trans[
            1, 1] + trans[1, 2]
    return pts_out

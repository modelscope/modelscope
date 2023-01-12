# Part of the implementation is borrowed and modified from DIFRINT,
# publicly available at https://github.com/jinsc37/DIFRINT/blob/master/metrics.py

import os
import sys
import tempfile
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm

from modelscope.metainfo import Metrics
from modelscope.models.cv.video_stabilization.utils.WarpUtils import \
    warpListImage
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.video_stabilization_metric)
class VideoStabilizationMetric(Metric):
    """The metric for video summarization task.
    """

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, outputs: Dict, inputs: Dict):
        out = video_merger(warpprocess(outputs))
        self.outputs.append(out['video'])
        self.inputs.append(inputs['input'][0])

    def evaluate(self):
        CR = []
        DV = []
        SS = []
        for output, input in zip(self.outputs, self.inputs):
            cropping_ratio, distortion_value, stability_score = \
                metrics(input, output)
            if cropping_ratio <= 1 and distortion_value <= 1 and stability_score <= 1:
                CR.append(cropping_ratio)
                DV.append(distortion_value)
                SS.append(stability_score)
            else:
                print('Removed one error item when computing metrics.')

        return {
            MetricKeys.CROPPING_RATIO: sum(CR) / len(CR),
            MetricKeys.DISTORTION_VALUE: sum(DV) / len(DV),
            MetricKeys.STABILITY_SCORE: sum(SS) / len(SS),
        }

    def merge(self, other: 'VideoStabilizationMetric'):
        self.inputs.extend(other.inputs)
        self.outputs.extend(other.outputs)

    def __getstate__(self):
        return self.inputs, self.outputs

    def __setstate__(self, state):
        self.inputs, self.outputs = state


def warpprocess(inputs):
    """ video stabilization postprocess

    Args:
        inputs:  input data

    Return:
        dict of results:  a dict containing outputs of model.
    """
    x_paths = inputs['origin_motion'][:, :, :, 0]
    y_paths = inputs['origin_motion'][:, :, :, 1]
    sx_paths = inputs['smooth_path'][:, :, :, 0]
    sy_paths = inputs['smooth_path'][:, :, :, 1]
    new_x_motion_meshes = sx_paths - x_paths
    new_y_motion_meshes = sy_paths - y_paths
    out_images = warpListImage(inputs['ori_images'], new_x_motion_meshes,
                               new_y_motion_meshes, inputs['width'],
                               inputs['height'])

    return {
        'output': out_images,
        'fps': inputs['fps'],
        'width': inputs['width'],
        'height': inputs['height'],
        'base_crop_width': inputs['base_crop_width']
    }


def video_merger(inputs):
    out_images = inputs['output'].numpy().astype(np.uint8)
    out_images = [
        np.transpose(out_images[idx], (1, 2, 0))
        for idx in range(out_images.shape[0])
    ]

    output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = inputs['width']
    h = inputs['height']
    base_crop_width = inputs['base_crop_width']
    video_writer = cv2.VideoWriter(output_video_path, fourcc, inputs['fps'],
                                   (w, h))

    for idx, frame in enumerate(out_images):
        horizontal_border = int(base_crop_width * w / 1280)
        vertical_border = int(horizontal_border * h / w)
        new_frame = frame[vertical_border:-vertical_border,
                          horizontal_border:-horizontal_border]
        new_frame = cv2.resize(new_frame, (w, h))
        video_writer.write(new_frame)
    video_writer.release()

    return {'video': output_video_path}


def metrics(original_v, pred_v):
    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    sift = cv2.SIFT_create()

    # Apply the homography transformation if we have enough good matches
    MIN_MATCH_COUNT = 10

    ratio = 0.7
    thresh = 5.0

    CR_seq = []
    DV_seq = []
    Pt = np.eye(3)
    P_seq = []

    vc_o = cv2.VideoCapture(original_v)
    vc_p = cv2.VideoCapture(pred_v)

    rval_o = vc_o.isOpened()
    rval_p = vc_p.isOpened()

    imgs1 = []
    imgs1o = []
    while (rval_o and rval_p):
        rval_o, img1 = vc_o.read()
        rval_p, img1o = vc_p.read()
        if rval_o and rval_p:
            imgs1.append(img1)
            imgs1o.append(img1o)
    is_got_bad_item = False
    print('processing ' + original_v.split('/')[-1] + ':')
    for i in tqdm(range(len(imgs1))):
        # Load the images in gray scale
        img1 = imgs1[i]
        img1o = imgs1o[i]

        # Detect the SIFT key points and compute the descriptors for the two images
        keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keyPoints1o, descriptors1o = sift.detectAndCompute(img1o, None)

        # Match the descriptors
        matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

        # Select the good matches using the ratio test
        goodMatches = []

        for m, n in matches:
            if m.distance < ratio * n.distance:
                goodMatches.append(m)

        if len(goodMatches) > MIN_MATCH_COUNT:
            # Get the good key points positions
            sourcePoints = np.float32([
                keyPoints1[m.queryIdx].pt for m in goodMatches
            ]).reshape(-1, 1, 2)
            destinationPoints = np.float32([
                keyPoints1o[m.trainIdx].pt for m in goodMatches
            ]).reshape(-1, 1, 2)

            # Obtain the homography matrix
            M, _ = cv2.findHomography(
                sourcePoints,
                destinationPoints,
                method=cv2.RANSAC,
                ransacReprojThreshold=thresh)
        else:
            is_got_bad_item = True

        # end

        if not is_got_bad_item:
            # Obtain Scale, Translation, Rotation, Distortion value
            # Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
            scaleRecovered = np.sqrt(M[0, 1]**2 + M[0, 0]**2)

            w, _ = np.linalg.eig(M[0:2, 0:2])
            w = np.sort(w)[::-1]
            DV = w[1] / w[0]

            CR_seq.append(1 / scaleRecovered)
            DV_seq.append(DV)

            # For Stability score calculation
            if i + 1 < len(imgs1):
                img2o = imgs1o[i + 1]

                keyPoints2o, descriptors2o = sift.detectAndCompute(img2o, None)
                matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)
                goodMatches = []

                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        goodMatches.append(m)

                if len(goodMatches) > MIN_MATCH_COUNT:
                    # Get the good key points positions
                    sourcePoints = np.float32([
                        keyPoints1o[m.queryIdx].pt for m in goodMatches
                    ]).reshape(-1, 1, 2)
                    destinationPoints = np.float32([
                        keyPoints2o[m.trainIdx].pt for m in goodMatches
                    ]).reshape(-1, 1, 2)

                    # Obtain the homography matrix
                    M, _ = cv2.findHomography(
                        sourcePoints,
                        destinationPoints,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=thresh)
                # end

                P_seq.append(np.matmul(Pt, M))
                Pt = np.matmul(Pt, M)
            # end
    # end

    if is_got_bad_item:
        return -1, -1, -1

    # Make 1D temporal signals
    P_seq_t = []
    P_seq_r = []

    for Mp in P_seq:
        transRecovered = np.sqrt(Mp[0, 2]**2 + Mp[1, 2]**2)
        # Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
        thetaRecovered = np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi
        P_seq_t.append(transRecovered)
        P_seq_r.append(thetaRecovered)

    # FFT
    fft_t = np.fft.fft(P_seq_t)
    fft_r = np.fft.fft(P_seq_r)
    fft_t = np.abs(fft_t)**2
    fft_r = np.abs(fft_r)**2

    fft_t = np.delete(fft_t, 0)
    fft_r = np.delete(fft_r, 0)
    fft_t = fft_t[:len(fft_t) // 2]
    fft_r = fft_r[:len(fft_r) // 2]

    SS_t = np.sum(fft_t[:5]) / np.sum(fft_t)
    SS_r = np.sum(fft_r[:5]) / np.sum(fft_r)

    return np.min([np.mean(CR_seq),
                   1]), np.absolute(np.min(DV_seq)), (SS_t + SS_r) / 2

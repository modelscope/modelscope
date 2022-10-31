# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math
from typing import Any

import cv2
import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from .base import EasyCVPipeline


@PIPELINES.register_module(
    Tasks.face_2d_keypoints, module_name=Pipelines.face_2d_keypoints)
class Face2DKeypointsPipeline(EasyCVPipeline):
    """Pipeline for face 2d keypoints detection."""

    def __init__(self,
                 model: str,
                 model_file_pattern=ModelFile.TORCH_MODEL_FILE,
                 *args,
                 **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """

        super(Face2DKeypointsPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)

        # face detect pipeline
        det_model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'
        self.face_detection = pipeline(
            Tasks.face_detection, model=det_model_id)

    def show_result(self, img, points, scale=2, save_path=None):
        return self.predict_op.show_result(img, points, scale, save_path)

    def _choose_face(self, det_result, min_face=10):
        """
        choose face with maximum area
        Args:
            det_result: output of face detection pipeline
            min_face: minimum size of valid face w/h
        """
        bboxes = np.array(det_result[OutputKeys.BOXES])
        landmarks = np.array(det_result[OutputKeys.KEYPOINTS])
        if bboxes.shape[0] == 0:
            logger.warn('No face detected!')
            return None
        # face idx with enough size
        face_idx = []
        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            if (box[2] - box[0]) >= min_face and (box[3] - box[1]) >= min_face:
                face_idx += [i]
        if len(face_idx) == 0:
            logger.warn(
                f'Face size not enough, less than {min_face}x{min_face}!')
            return None
        bboxes = bboxes[face_idx]
        landmarks = landmarks[face_idx]

        return bboxes, landmarks

    def expend_box(self, box, w, h, scalex=0.3, scaley=0.5):
        x1 = box[0]
        y1 = box[1]
        wb = box[2] - x1
        hb = box[3] - y1
        deltax = int(wb * scalex)
        deltay1 = int(hb * scaley)
        deltay2 = int(hb * scalex)
        x1 = x1 - deltax
        y1 = y1 - deltay1
        if x1 < 0:
            deltax = deltax + x1
            x1 = 0
        if y1 < 0:
            deltay1 = deltay1 + y1
            y1 = 0
        x2 = x1 + wb + 2 * deltax
        y2 = y1 + hb + deltay1 + deltay2
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        return [x1, y1, x2, y2]

    def rotate_point(self, angle, center, landmark):
        rad = angle * np.pi / 180.0
        alpha = np.cos(rad)
        beta = np.sin(rad)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0, 0] = alpha
        M[0, 1] = beta
        M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
        M[1, 0] = -beta
        M[1, 1] = alpha
        M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

        landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                                 M[1, 0] * x + M[1, 1] * y + M[1, 2])
                                for (x, y) in landmark])
        return M, landmark_

    def random_normal(self):
        """
        3-sigma rule
        return: (-1, +1)
        """
        mu, sigma = 0, 1
        while True:
            s = np.random.normal(mu, sigma)
            if s < mu - 3 * sigma or s > mu + 3 * sigma:
                continue
            return s / 3 * sigma

    def rotate_crop_img(self, img, pts, M):
        image_size = 256
        enlarge_ratio = 1.1

        imgT = cv2.warpAffine(img, M, (int(img.shape[1]), int(img.shape[0])))

        x1 = pts[5][0]
        y1 = pts[5][1]
        x2 = pts[6][0]
        y2 = pts[6][1]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        x1 = int(x1 - (enlarge_ratio - 1.0) / 2.0 * w)
        y1 = int(y1 - (enlarge_ratio - 1.0) / 2.0 * h)

        new_w = int(enlarge_ratio * (1 + self.random_normal() * 0.1) * w)
        new_h = int(enlarge_ratio * (1 + self.random_normal() * 0.1) * h)
        new_x1 = x1 + int(self.random_normal() * image_size * 0.05)
        new_y1 = y1 + int(self.random_normal() * image_size * 0.05)
        new_x2 = new_x1 + new_w
        new_y2 = new_y1 + new_h

        height, width, _ = imgT.shape
        dx = max(0, -new_x1)
        dy = max(0, -new_y1)
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)

        edx = max(0, new_x2 - width)
        edy = max(0, new_y2 - height)
        new_x2 = min(width, new_x2)
        new_y2 = min(height, new_y2)

        sub_imgT = imgT[new_y1:new_y2, new_x1:new_x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            sub_imgT = cv2.copyMakeBorder(
                sub_imgT,
                dy,
                edy,
                dx,
                edx,
                cv2.BORDER_CONSTANT,
                value=(103.94, 116.78, 123.68))

        return sub_imgT, imgT, [new_x1, new_y1, new_x2,
                                new_y2], [dx, dy, edx, edy]

    def crop_img(self, imgT, pts, angle):
        image_size = 256
        enlarge_ratio = 1.1

        x1 = np.min(pts[:, 0])
        x2 = np.max(pts[:, 0])
        y1 = np.min(pts[:, 1])
        y2 = np.max(pts[:, 1])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        x1 = int(x1 - (enlarge_ratio - 1.0) / 2.0 * w)
        y1 = int(y1 - (enlarge_ratio - 1.0) / 2.0 * h)

        new_w = int(enlarge_ratio * (1 + self.random_normal() * 0.1) * w)
        new_h = int(enlarge_ratio * (1 + self.random_normal() * 0.1) * h)
        new_x1 = x1 + int(self.random_normal() * image_size * 0.05)
        new_y1 = y1 + int(self.random_normal() * image_size * 0.05)
        new_x2 = new_x1 + new_w
        new_y2 = new_y1 + new_h

        new_xy = new_x1, new_y1
        pts = pts - new_xy

        height, width, _ = imgT.shape
        dx = max(0, -new_x1)
        dy = max(0, -new_y1)
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)

        edx = max(0, new_x2 - width)
        edy = max(0, new_y2 - height)
        new_x2 = min(width, new_x2)
        new_y2 = min(height, new_y2)

        sub_imgT = imgT[new_y1:new_y2, new_x1:new_x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            sub_imgT = cv2.copyMakeBorder(
                sub_imgT,
                dy,
                edy,
                dx,
                edx,
                cv2.BORDER_CONSTANT,
                value=(103.94, 116.78, 123.68))

        return sub_imgT, [new_x1, new_y1, new_x2, new_y2], [dx, dy, edx, edy]

    def __call__(self, inputs) -> Any:
        image_size = 256

        img = LoadImage.convert_to_ndarray(inputs)
        h, w, c = img.shape
        img_rgb = copy.deepcopy(img)
        img_rgb = img_rgb[:, :, ::-1]
        det_result = self.face_detection(img_rgb)
        boxes, keypoints = self._choose_face(det_result)

        output_boxes = []
        output_keypoints = []
        output_poses = []
        for idx, box_ori in enumerate(boxes):
            box = self.expend_box(box_ori, w, h, scalex=0.15, scaley=0.15)
            y0 = int(box[1])
            y1 = int(box[3])
            x0 = int(box[0])
            x1 = int(box[2])
            sub_img = img[y0:y1, x0:x1]

            keypoint = keypoints[idx]
            pts = [[keypoint[0], keypoint[1]], [keypoint[2], keypoint[3]],
                   [keypoint[4], keypoint[5]], [keypoint[6], keypoint[7]],
                   [keypoint[8], keypoint[9]], [box[0], box[1]],
                   [box[2], box[3]]]
            # radian
            angle = math.atan2((pts[1][1] - pts[0][1]),
                               (pts[1][0] - pts[0][0]))
            # angle
            theta = angle * (180 / np.pi)

            center = [image_size // 2, image_size // 2]
            cx, cy = center
            M, landmark_ = self.rotate_point(theta, (cx, cy), pts)
            sub_img, imgT, bbox, delta_border = self.rotate_crop_img(
                img, pts, M)

            outputs = self.predict_op([sub_img])[0]
            tmp_keypoints = outputs['point']

            for idx in range(0, len(tmp_keypoints)):
                tmp_keypoints[idx][0] += (delta_border[0] + bbox[0])
                tmp_keypoints[idx][1] += (delta_border[1] + bbox[1])

            for idx in range(0, 3):
                sub_img, bbox, delta_border = self.crop_img(
                    imgT, tmp_keypoints, 0)
                outputs = self.predict_op([sub_img])[0]
                tmp_keypoints = outputs['point']
                for idx in range(0, len(tmp_keypoints)):
                    tmp_keypoints[idx][0] += (delta_border[0] + bbox[0])
                    tmp_keypoints[idx][1] += (delta_border[1] + bbox[1])

            M2, tmp_keypoints = self.rotate_point(-theta, (cx, cy),
                                                  tmp_keypoints)

            output_keypoints.append(np.array(tmp_keypoints))
            output_poses.append(np.array(outputs['pose']))
            output_boxes.append(np.array(box_ori))

        results = {
            OutputKeys.KEYPOINTS: output_keypoints,
            OutputKeys.POSES: output_poses,
            OutputKeys.BOXES: output_boxes
        }

        return results

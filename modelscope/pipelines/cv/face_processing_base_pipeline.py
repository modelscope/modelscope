# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class FaceProcessingBasePipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face processing pipeline and output cropped img, scores, bbox and lmks.

        Args:
            model: model id on modelscope hub.

        """
        super().__init__(model=model, **kwargs)
        # face detect pipeline
        det_model_id = 'damo/cv_ddsar_face-detection_iclr23-damofd'
        self.face_detection = pipeline(
            Tasks.face_detection, model=det_model_id)

    def _choose_face(self,
                     det_result,
                     min_face=10,
                     top_face=1,
                     center_face=False,
                     img_shape=None):
        '''
        choose face with maximum area
        Args:
            det_result: output of face detection pipeline
            min_face: minimum size of valid face w/h
            top_face: take faces with top max areas
            center_face: choose the most centerd face from multi faces, only valid if top_face > 1
        '''
        bboxes = np.array(det_result[OutputKeys.BOXES])
        landmarks = np.array(det_result[OutputKeys.KEYPOINTS])
        scores = np.array(det_result[OutputKeys.SCORES])
        if bboxes.shape[0] == 0:
            logger.info('Warning: No face detected!')
            return None
        # face idx with enough size
        face_idx = []
        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            if (box[2] - box[0]) >= min_face and (box[3] - box[1]) >= min_face:
                face_idx += [i]
        if len(face_idx) == 0:
            logger.info(
                f'Warning: Face size not enough, less than {min_face}x{min_face}!'
            )
            return None
        bboxes = bboxes[face_idx]
        landmarks = landmarks[face_idx]
        scores = scores[face_idx]
        # find max faces
        boxes = np.array(bboxes)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sort_idx = np.argsort(area)[-top_face:]
        # find center face
        if top_face > 1 and center_face and bboxes.shape[0] > 1 and img_shape:
            img_center = [img_shape[1] // 2, img_shape[0] // 2]
            min_dist = float('inf')
            sel_idx = -1
            for _idx in sort_idx:
                box = boxes[_idx]
                dist = np.square(
                    np.abs((box[0] + box[2]) / 2 - img_center[0])) + np.square(
                        np.abs((box[1] + box[3]) / 2 - img_center[1]))
                if dist < min_dist:
                    min_dist = dist
                    sel_idx = _idx
            sort_idx = [sel_idx]
        main_idx = sort_idx[-1]
        return scores[main_idx], bboxes[main_idx], landmarks[main_idx]

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = img[:, :, ::-1]
        det_result = self.face_detection(img.copy())
        rtn = self._choose_face(det_result, img_shape=img.shape)
        if rtn is not None:
            scores, bboxes, face_lmks = rtn
            face_lmks = face_lmks.reshape(5, 2)
            align_img, _ = align_face(img, (112, 112), face_lmks)

            result = {}
            result['img'] = np.ascontiguousarray(align_img)
            result['scores'] = [scores]
            result['bbox'] = bboxes
            result['lmks'] = face_lmks
            return result
        else:
            return None

    def align_face_padding(self, img, rect, padding_size=16, pad_pixel=127):
        rect = np.reshape(rect, (-1, 4))
        if img is None:
            return None
        if img.ndim == 2:
            w, h = img.shape
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
            img = ret
        img = img[:, :, 0:3]
        img = img[..., ::-1]
        nrof = np.zeros((5, ), dtype=np.int32)

        bounding_boxes = rect
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]
            bindex = 0
            if nrof_faces > 1:
                img_center = img_size / 2
                offsets = np.vstack([
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]
                ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                bindex = np.argmax(0 - offset_dist_squared * 2.0)
            _bbox = bounding_boxes[bindex, 0:4]
            nrof[0] += 1
        else:
            nrof[1] += 1
        if _bbox is None:
            nrof[2] += 1
            return None
        _bbox = [int(_bbox[0]), int(_bbox[1]), int(_bbox[2]), int(_bbox[3])]
        x1 = _bbox[0] - int(
            (_bbox[2] - _bbox[0] + 1) * padding_size * 1.0 / 112)
        x2 = _bbox[2] + int(
            (_bbox[2] - _bbox[0] + 1) * padding_size * 1.0 / 112)
        y1 = _bbox[1] - int(
            (_bbox[3] - _bbox[1] + 1) * padding_size * 1.0 / 112)
        y2 = _bbox[3] + int(
            (_bbox[3] - _bbox[1] + 1) * padding_size * 1.0 / 112)
        _bbox[0] = max(0, x1)
        _bbox[1] = max(0, y1)
        _bbox[2] = min(img.shape[1] - 1, x2)
        _bbox[3] = min(img.shape[0] - 1, y2)
        padding_h = _bbox[3] - _bbox[1] + 1
        padding_w = _bbox[2] - _bbox[0] + 1
        if padding_w > padding_h:
            offset = int((padding_w - padding_h) / 2)
            _bbox[1] = _bbox[1] - offset
            _bbox[3] = _bbox[1] + padding_w - 1
            _bbox[1] = max(0, _bbox[1])
            _bbox[3] = min(img.shape[0] - 1, _bbox[3])
            dst_size = padding_w
        else:
            offset = int((padding_h - padding_w) / 2)
            _bbox[0] = _bbox[0] - offset
            _bbox[2] = _bbox[0] + padding_h - 1
            _bbox[0] = max(0, _bbox[0])
            _bbox[2] = min(img.shape[1] - 1, _bbox[2])
            dst_size = padding_h

        dst = np.full((dst_size, dst_size, 3), pad_pixel, dtype=np.uint8)
        dst_x_offset = int((dst_size - (_bbox[2] - _bbox[0] + 1)) / 2)
        dst_y_offset = int((dst_size - (_bbox[3] - _bbox[1] + 1)) / 2)

        y_start = dst_y_offset
        y_end = dst_y_offset + _bbox[3] + 1 - _bbox[1]
        x_start = dst_x_offset
        x_end = dst_x_offset + _bbox[2] + 1 - _bbox[0]
        dst[y_start:y_end, x_start:x_end, :] = img[_bbox[1]:_bbox[3] + 1,
                                                   _bbox[0]:_bbox[2] + 1, :]

        dst = cv2.resize(dst, (128, 128), interpolation=cv2.INTER_LINEAR)

        return dst

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            OutputKeys.OUTPUT_IMG: input['img'].cpu().numpy(),
            OutputKeys.SCORES: input['scores'].cpu().tolist(),
            OutputKeys.BOXES: [input['bbox'].cpu().tolist()],
            OutputKeys.KEYPOINTS: [input['lmks'].cpu().tolist()]
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

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
        det_model_id = 'damo/cv_resnet50_face-detection_retinaface'
        self.face_detection = pipeline(
            Tasks.face_detection, model=det_model_id)

    def _choose_face(self,
                     det_result,
                     min_face=10,
                     top_face=1,
                     center_face=False):
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
        if top_face > 1 and center_face and bboxes.shape[0] > 1:
            img_center = [img.shape[1] // 2, img.shape[0] // 2]
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
        rtn = self._choose_face(det_result)
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

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            OutputKeys.OUTPUT_IMG: input['img'].cpu().numpy(),
            OutputKeys.SCORES: input['scores'].cpu().tolist(),
            OutputKeys.BOXES: [input['bbox'].cpu().tolist()],
            OutputKeys.KEYPOINTS: [input['lmks'].cpu().tolist()]
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

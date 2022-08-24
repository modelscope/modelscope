import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.models.cv.face_recognition.torchkit.backbone import get_model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_recognition, module_name=Pipelines.face_recognition)
class FaceRecognitionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        # face recong model
        super().__init__(model=model, **kwargs)
        device = torch.device(
            f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        self.device = device
        face_model = get_model('IR_101')([112, 112])
        face_model.load_state_dict(
            torch.load(
                osp.join(model, ModelFile.TORCH_MODEL_BIN_FILE),
                map_location=device))
        face_model = face_model.to(device)
        face_model.eval()
        self.face_model = face_model
        logger.info('face recognition model loaded!')
        # face detect pipeline
        det_model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'
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
        # scores = np.array(det_result[OutputKeys.SCORES])
        if bboxes.shape[0] == 0:
            logger.info('No face detected!')
            return None
        # face idx with enough size
        face_idx = []
        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            if (box[2] - box[0]) >= min_face and (box[3] - box[1]) >= min_face:
                face_idx += [i]
        if len(face_idx) == 0:
            logger.info(
                f'Face size not enough, less than {min_face}x{min_face}!')
            return None
        bboxes = bboxes[face_idx]
        landmarks = landmarks[face_idx]
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
        return bboxes[main_idx], landmarks[main_idx]

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = img[:, :, ::-1]
        det_result = self.face_detection(img.copy())
        rtn = self._choose_face(det_result)
        face_img = None
        if rtn is not None:
            _, face_lmks = rtn
            face_lmks = face_lmks.reshape(5, 2)
            align_img, _ = align_face(img, (112, 112), face_lmks)
            face_img = align_img[:, :, ::-1]  # to rgb
            face_img = np.transpose(face_img, axes=(2, 0, 1))
            face_img = (face_img / 255. - 0.5) / 0.5
            face_img = face_img.astype(np.float32)
        result = {}
        result['img'] = face_img
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        assert input['img'] is not None
        img = input['img'].unsqueeze(0)
        emb = self.face_model(img).detach().cpu().numpy()
        emb /= np.sqrt(np.sum(emb**2, -1, keepdims=True))  # l2 norm
        return {OutputKeys.IMG_EMBEDDING: emb}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

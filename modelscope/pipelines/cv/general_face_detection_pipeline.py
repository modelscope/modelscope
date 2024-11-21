# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import numpy as np
import torch
import cv2
import copy
import argparse
import os

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.models.cv.general_face_detection import infer  
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

@PIPELINES.register_module(
    Tasks.general_face_detection, module_name=Pipelines.general_face_detection)
class GeneralFaceDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth prediction pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        model_path = os.path.join(model, 'det_model.pth')
        # model_path = '/home/gyalex/projects/develop/Pytorch_Retinaface/weights/Resnet50_epoch_50.pth'
        self.net, self.device, self.args = infer.init_model(model_path)

        logger.info('General face detection model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        print('start preprocess')

        if isinstance(input, str):
            # image = LoadImage.convert_to_ndarray(input)
            image = cv2.imread(input)
        elif isinstance(input, np.ndarray):
            image = copy.deepcopy(input)

        data = {'image': image}

        self.width  = image.shape[1]
        self.height = image.shape[0]

        print('finish preprocess')
        
        return data
    
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        print('start infer')
        
        image = input['image']
        results = infer.process(self.net, image, self.device, self.args)
        
        print('finish infer')
        
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        all_faces = inputs[:,0:5]
        for num in range(all_faces.shape[0]):
            if all_faces[num,0] < 0:
                all_faces[num,0] = 0
            if all_faces[num,1] < 0:
                all_faces[num,1] = 0
            if all_faces[num,2] >= self.width:
                all_faces[num,2] = self.width-1
            if all_faces[num,3] >= self.height:
                all_faces[num,3] = self.height-1

        outputs = {'faces': all_faces}
        return outputs

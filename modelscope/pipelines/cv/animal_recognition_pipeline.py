# Copyright 2021-2022 The Alibaba Fundamental Vision  Team Authors. All rights reserved.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.animal_recognition import Bottleneck, ResNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Devices, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.animal_recognition, module_name=Pipelines.animal_recognition)
class AnimalRecognitionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a animal recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        import torch

        def resnest101(**kwargs):
            model = ResNet(
                Bottleneck, [3, 4, 23, 3],
                radix=2,
                groups=1,
                bottleneck_width=64,
                deep_stem=True,
                stem_width=64,
                avg_down=True,
                avd=True,
                avd_first=False,
                **kwargs)
            return model

        def filter_param(src_params, own_state):
            copied_keys = []
            for name, param in src_params.items():
                if 'module.' == name[0:7]:
                    name = name[7:]
                if '.module.' not in list(own_state.keys())[0]:
                    name = name.replace('.module.', '.')
                if (name in own_state) and (own_state[name].shape
                                            == param.shape):
                    own_state[name].copy_(param)
                    copied_keys.append(name)

        def load_pretrained(model, src_params):
            if 'state_dict' in src_params:
                src_params = src_params['state_dict']
            own_state = model.state_dict()
            filter_param(src_params, own_state)
            model.load_state_dict(own_state)

        self.local_path = self.model
        src_params = torch.load(
            osp.join(self.local_path, ModelFile.TORCH_MODEL_FILE), Devices.cpu)
        self.model = resnest101(num_classes=8288)
        load_pretrained(self.model, src_params)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize
        ])
        img = test_transforms(img)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        def set_phase(model, is_train):
            if is_train:
                model.train()
            else:
                model.eval()

        is_train = False
        set_phase(self.model, is_train)
        img = input['img']
        input_img = torch.unsqueeze(img, 0)
        outputs = self.model(input_img)
        return {'outputs': outputs}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        label_mapping_path = osp.join(self.local_path, 'label_mapping.txt')
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = f.readlines()
        score = torch.max(inputs['outputs'])
        inputs = {
            OutputKeys.SCORES: [score.item()],
            OutputKeys.LABELS:
            [label_mapping[inputs['outputs'].argmax()].split('\t')[1]]
        }
        return inputs

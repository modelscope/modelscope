# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os.path as osp
from typing import Any, Dict

import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.tinynas_classfication import get_zennet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_classification, module_name=Pipelines.tinynas_classification)
class TinynasClassificationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a tinynas classification pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.path = model
        self.model = get_zennet()

        model_pth_path = osp.join(self.path, ModelFile.TORCH_MODEL_FILE)

        checkpoint = torch.load(model_pth_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)

        input_image_size = 224
        crop_image_size = 380
        input_image_crop = 0.875
        resize_image_size = int(math.ceil(crop_image_size / input_image_crop))
        transforms_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_list = [
            transforms.Resize(
                resize_image_size,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_image_size),
            transforms.ToTensor(), transforms_normalize
        ]
        transformer = transforms.Compose(transform_list)

        img = transformer(img)
        img = torch.unsqueeze(img, 0)
        img = torch.nn.functional.interpolate(
            img, input_image_size, mode='bilinear')
        result = {'img': img}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        is_train = False
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        outputs = self.model(input['img'])
        return {'outputs': outputs}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        label_mapping_path = osp.join(self.path, 'label_map.txt')
        f = open(label_mapping_path, encoding='utf-8')
        content = f.read()
        f.close()
        label_dict = eval(content)

        output_prob = torch.nn.functional.softmax(inputs['outputs'], dim=-1)
        score = torch.max(output_prob)
        output_dict = {
            OutputKeys.SCORES: [score.item()],
            OutputKeys.LABELS: [label_dict[inputs['outputs'].argmax().item()]]
        }
        return output_dict

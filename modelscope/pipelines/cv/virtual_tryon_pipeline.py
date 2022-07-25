# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Pipelines
from modelscope.models.cv.virual_tryon.sdafnet import SDAFNet_Tryon
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import load_image
from modelscope.utils.constant import ModelFile, Tasks
from ..base import Pipeline
from ..builder import PIPELINES


@PIPELINES.register_module(
    Tasks.virtual_tryon, module_name=Pipelines.virtual_tryon)
class VirtualTryonPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model`  to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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

        self.model = SDAFNet_Tryon(ref_in_channel=6).to(self.device)
        local_model_dir = model
        if osp.exists(model):
            local_model_dir = model
        else:
            local_model_dir = snapshot_download(model)
        self.local_path = local_model_dir
        src_params = torch.load(
            osp.join(local_model_dir, ModelFile.TORCH_MODEL_FILE), 'cpu')
        load_pretrained(self.model, src_params)
        self.model = self.model.eval()
        self.size = 192
        from torchvision import transforms
        self.test_transforms = transforms.Compose([
            transforms.Resize(self.size, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(input['masked_model'], str):
            img_agnostic = load_image(input['masked_model'])
            pose = load_image(input['pose'])
            cloth_img = load_image(input['cloth'])
        elif isinstance(input['masked_model'], PIL.Image.Image):
            img_agnostic = img_agnostic.convert('RGB')
            pose = pose.convert('RGB')
            cloth_img = cloth_img.convert('RGB')
        elif isinstance(input['masked_model'], np.ndarray):
            if len(input.shape) == 2:
                img_agnostic = cv2.cvtColor(img_agnostic, cv2.COLOR_GRAY2BGR)
                pose = cv2.cvtColor(pose, cv2.COLOR_GRAY2BGR)
                cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_GRAY2BGR)
            img_agnostic = Image.fromarray(
                img_agnostic[:, :, ::-1].astype('uint8')).convert('RGB')
            pose = Image.fromarray(
                pose[:, :, ::-1].astype('uint8')).convert('RGB')
            cloth_img = Image.fromarray(
                cloth_img[:, :, ::-1].astype('uint8')).convert('RGB')
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')

        img_agnostic = self.test_transforms(img_agnostic)
        pose = self.test_transforms(pose)
        cloth_img = self.test_transforms(cloth_img)
        inputs = {
            'masked_model': img_agnostic.unsqueeze(0),
            'pose': pose.unsqueeze(0),
            'cloth': cloth_img.unsqueeze(0)
        }
        return inputs

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        img_agnostic = inputs['masked_model'].to(self.device)
        pose = inputs['pose'].to(self.device)
        cloth_img = inputs['cloth'].to(self.device)
        ref_input = torch.cat((pose, img_agnostic), dim=1)
        tryon_result = self.model(ref_input, cloth_img, img_agnostic)
        return {OutputKeys.OUTPUT_IMG: tryon_result}

    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        tryon_result = outputs[OutputKeys.OUTPUT_IMG].permute(0, 2, 3,
                                                              1).squeeze(0)
        tryon_result = tryon_result.add(1.).div(2.).mul(255).data.cpu().numpy()
        outputs[OutputKeys.OUTPUT_IMG] = tryon_result
        return outputs

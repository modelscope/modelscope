# Copyright © Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from modelscope.models import Model
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks, ModelFile
from modelscope.preprocessors import Preprocessor


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionPipeline(Pipeline):

    def __init__(self, model: str, 
                 device: str = 'gpu',
                 config_file: str = None, 
                 preprocessor: Optional[Preprocessor] = None,
                 auto_collate=True,
                 **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub or local model dir.
        """
        super().__init__(model=model,
                         device=device,
                         config_file=config_file,
                         preprocessor=preprocessor,
                         auto_collate=auto_collate)
        
        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        
        # torch_dtype = kwargs.get('torch_dtype', torch.float32)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = self.model.to(self.device)
        # self.model.eval()
        self.preprocessor = transforms.Compose([
            transforms.Resize(
                512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            results = self.model(**inputs)
            return results

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}

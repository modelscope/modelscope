from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.super_resolution import RRDBNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_super_resolution, module_name=Pipelines.image_super_resolution)
class ImageSuperResolutionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image super resolution pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.device = 'cpu'
        self.num_feat = 64
        self.num_block = 23
        self.scale = 4
        self.sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=self.num_feat,
            num_block=self.num_block,
            num_grow_ch=32,
            scale=self.scale).to(self.device)

        model_path = f'{self.model}/{ModelFile.TORCH_MODEL_FILE}'
        self.sr_model.load_state_dict(torch.load(model_path), strict=True)

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = torch.from_numpy(img).to(self.device).permute(
            2, 0, 1).unsqueeze(0) / 255.
        result = {'img': img}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        self.sr_model.eval()
        with torch.no_grad():
            out = self.sr_model(input['img'])

        out = out.squeeze(0).permute(1, 2, 0).flip(2)
        out_img = np.clip(out.float().cpu().numpy(), 0, 1) * 255

        return {OutputKeys.OUTPUT_IMG: out_img.astype(np.uint8)}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

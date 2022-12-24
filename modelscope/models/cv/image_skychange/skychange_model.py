import math
import os
import pdb
import time
from collections import OrderedDict
from typing import Any, Dict, List, Union

import cv2
import json
import torch
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .ptsemseg.hrnet_super_and_ocr import HrnetSuperAndOcr
from .ptsemseg.unet import Unet
from .skychange import blend

logger = get_logger()


@MODELS.register_module(
    Tasks.image_skychange, module_name=Models.image_skychange)
class ImageSkychange(TorchModel):

    def __init__(self, model_dir, refine_cfg, coarse_cfg, *args, **kwargs):
        """
        Args:
            model_dir (str): model directory to initialize some resource.
            refine_cfg: configuration of refine model.
            coarse_cfg: configuration of coarse model.
        """
        super().__init__(model_dir=model_dir, *args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info('Use GPU: {}'.format(self.device))
        else:
            self.device = torch.device('cpu')
            logger.info('Use CPU: {}'.format(self.device))

        coarse_model_path = '{}/{}'.format(model_dir,
                                           ModelFile.TORCH_MODEL_FILE)
        refine_model_path = '{}/{}'.format(model_dir,
                                           'unet_sky_matting_final_model.pkl')

        logger.info(
            '####################### load refine models ################################'
        )
        self.refine_model = Unet(**refine_cfg['Model'])
        self.load_model(self.refine_model, refine_model_path)
        self.refine_model.eval()
        logger.info(
            '####################### load refine models done ############################'
        )

        logger.info(
            '####################### load coarse models ################################'
        )
        self.coarse_model = HrnetSuperAndOcr(**coarse_cfg['Model'])
        self.load_model(self.coarse_model, coarse_model_path)
        self.coarse_model.eval()
        logger.info(
            '####################### load coarse models done ############################'
        )

    def load_model(self, seg_model, input_model_path):
        if not os.path.isfile(input_model_path):
            logger.error(
                '[checkModelPath]:model path dose not exits!!! model Path:'
                + input_model_path)
            raise Exception('[checkModelPath]:model path dose not exits!')

        if torch.cuda.is_available():
            checkpoint = torch.load(input_model_path)
            model_state = self.convert_state_dict(checkpoint['model_state'])
            seg_model.load_state_dict(model_state)
            seg_model.to(self.device)
        else:
            checkpoint = torch.load(input_model_path, map_location='cpu')
            model_state = self.convert_state_dict(checkpoint['model_state'])
            seg_model.load_state_dict(model_state)

    def convert_state_dict(self, state_dict):
        """Converts a state dict saved from a dataParallel module to normal
        module state_dict inplace
        :param state_dict is the loaded DataParallel model_state
        """
        if not next(iter(state_dict)).startswith('module.'):
            return state_dict  # abort if dict is not a DataParallel model_state
        new_state_dict = OrderedDict()

        split_index = 0
        for cur_key, cur_value in state_dict.items():
            if cur_key.startswith('module.model'):
                split_index = 13
            elif cur_key.startswith('module'):
                split_index = 7

            break

        for k, v in state_dict.items():
            name = k[split_index:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def forward(
        self,
        sky_image: torch.Tensor,
        sky_image_refine: torch.Tensor,
        scene_image: torch.Tensor,
        scene_image_refine: torch.Tensor,
        img_metas: Dict[str, Any],
    ):
        """
        Args:
            sky_image (`torch.Tensor`): batched image tensor, shape is [1, 3, h', w'].
            sky_image_refine (`torch.Tensor`): batched image tensor, shape is [1, 3, refine_net_h, refine_net_w].
            scene_image (`torch.Tensor`): batched image tensor, shape is [1, 3, h, w].
            scene_image_refine (`torch.Tensor`): batched image tensor, shape is [1, 3, refine_net_h, refine_net_w].
            img_metas (`Dict[str, Any]`): image meta info.
        Return:
            `IMAGE: shape is [h, w, 3] (0~255)`
        """
        start = time.time()
        sky_img_metas, scene_img_metas, input_size = img_metas[
            'sky_img_metas'], img_metas['scene_img_metas'], img_metas[
                'input_size']
        sky_mask = self.inference_mask(sky_image_refine, sky_img_metas,
                                       input_size)
        scene_mask = self.inference_mask(scene_image_refine, scene_img_metas,
                                         input_size)
        end = time.time()
        logger.info(
            'Time of inferencing mask of sky and scene images:{}'.format(
                end - start))
        start = time.time()
        scene_mask = scene_mask * 255
        sky_mask = sky_mask * 255
        res = blend(scene_image, scene_mask, sky_image, sky_mask)
        end = time.time()
        logger.info('Time of blending: {}'.format(end - start))
        return res

    @torch.no_grad()
    def inference_mask(self, img, img_metas, input_size):
        self.eval()
        raw_h, raw_w = img_metas['ori_shape']
        pad_direction = img_metas['pad_direction']
        coarse_input_size = input_size['coarse_input_size']
        refine_input_size = input_size['refine_input_size']
        h, w = img_metas['refine_shape']
        resize_images = F.interpolate(
            img, coarse_input_size, mode='bilinear', align_corners=True)
        # get coarse result
        pred_scores = self.coarse_model(resize_images)
        if isinstance(pred_scores, (tuple, list)):
            pred_scores = pred_scores[-1]
        score = F.interpolate(
            input=pred_scores,
            size=refine_input_size,
            mode='bilinear',
            align_corners=True,
        )
        _, coarse_pred = torch.max(score, dim=1)  # [B, h, w]
        coarse_pred = coarse_pred.unsqueeze(1).type(img.dtype)
        img = torch.cat([img, coarse_pred], dim=1)  # [B, c=4, h, w]
        del resize_images
        del pred_scores
        del score
        del coarse_pred
        torch.cuda.empty_cache()
        cur_scores = self.refine_model(img)
        del img
        torch.cuda.empty_cache()
        cur_scores = torch.clip(cur_scores, 0, 1)
        cur_scores = cur_scores.detach().cpu().numpy()[0]

        # resize if cur_scores shape are not compatible with origin image shape
        ph, pw = cur_scores.shape
        if ph != h or pw != w:
            cur_scores = F.interpolate(
                input=cur_scores,
                size=(h, w),
                mode='nearest',
                align_corners=True)
        # unpad to get valid area and resize to origin size
        valid_cur_pred = cur_scores[pad_direction[1]:refine_input_size[0]
                                    - pad_direction[3],
                                    pad_direction[0]:refine_input_size[1]
                                    - pad_direction[2], ]
        valid_cur_pred = cv2.resize(valid_cur_pred, (raw_w, raw_h))
        del cur_scores
        torch.cuda.empty_cache()
        print('get refine mask done')
        return valid_cur_pred

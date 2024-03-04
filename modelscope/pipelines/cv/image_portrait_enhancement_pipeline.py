# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_portrait_enhancement import gpen
from modelscope.models.cv.image_portrait_enhancement.align_faces import (
    get_reference_facial_points, warp_and_crop_face)
from modelscope.models.cv.image_portrait_enhancement.eqface import fqa
from modelscope.models.cv.image_portrait_enhancement.retinaface import \
    detection
from modelscope.models.cv.super_resolution import rrdbnet_arch
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage, load_image
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_portrait_enhancement,
    module_name=Pipelines.image_portrait_enhancement)
class ImagePortraitEnhancementPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.use_sr = True

        self.size = 512
        if 'hires' in model:
            self.size = 1024
        self.n_mlp = 8
        self.channel_multiplier = 2
        self.narrow = 1
        self.face_enhancer = gpen.FullGenerator(
            self.size,
            512,
            self.n_mlp,
            self.channel_multiplier,
            narrow=self.narrow).to(self.device)

        gpen_model_path = f'{model}/{ModelFile.TORCH_MODEL_FILE}'
        self.face_enhancer.load_state_dict(
            torch.load(gpen_model_path, map_location=torch.device('cpu')),
            strict=True)

        logger.info('load face enhancer model done')

        self.threshold = 0.9
        detector_model_path = f'{model}/face_detection/RetinaFace-R50.pth'
        self.face_detector = detection.RetinaFaceDetection(
            detector_model_path, self.device)

        logger.info('load face detector model done')

        self.num_feat = 32
        self.num_block = 23
        self.scale = 2
        self.sr_model = rrdbnet_arch.RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=self.num_feat,
            num_block=self.num_block,
            num_grow_ch=32,
            scale=self.scale).to(self.device)

        sr_model_path = f'{model}/super_resolution/realesrnet_x{self.scale}.pth'
        self.sr_model.load_state_dict(
            torch.load(sr_model_path,
                       map_location=torch.device('cpu'))['params_ema'],
            strict=True)

        logger.info('load sr model done')

        self.fqa_thres = 0.1
        self.id_thres = 0.15
        self.alpha = 1.0
        backbone_model_path = f'{model}/face_quality/eqface_backbone.pth'
        fqa_model_path = f'{model}/face_quality/eqface_quality.pth'
        self.eqface = fqa.FQA(backbone_model_path, fqa_model_path, self.device)

        logger.info('load fqa model done')

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512, 3), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1,
                      cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

    def enhance_face(self, img):
        img = cv2.resize(img, (self.size, self.size))
        img_t = self.img2tensor(img)

        self.face_enhancer.eval()
        with torch.no_grad():
            out, __ = self.face_enhancer(img_t)
        del img_t

        out = self.tensor2img(out)

        return out

    def img2tensor(self, img, is_norm=True):
        img_t = torch.from_numpy(img).to(self.device) / 255.
        if is_norm:
            img_t = (img_t - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        return img_t

    def tensor2img(self, img_t, pmax=255.0, is_denorm=True, imtype=np.uint8):
        if is_denorm:
            img_t = img_t * 0.5 + 0.5
        img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2)  # RGB->BGR
        img_np = np.clip(img_t.float().cpu().numpy(), 0, 1) * pmax

        return img_np.astype(imtype)

    def sr_process(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')

        self.sr_model.eval()
        with torch.no_grad():
            output = self.sr_model(img)
            del img
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

        return output

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)

        if self.use_sr:
            img_sr = self.sr_process(img)
            img = cv2.resize(img, img_sr.shape[:2][::-1])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_sr = img.copy()

        result = {'img': img, 'img_sr': img_sr}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        img, img_sr = input['img'], input['img_sr']
        img, img_sr = img.cpu().numpy(), img_sr.cpu().numpy()
        facebs, landms = self.face_detector.detect(img)

        height, width = img.shape[:2]
        full_mask = np.zeros(img.shape, dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue

            facial5points = np.reshape(facial5points, (2, 5))

            of, of_112, tfm_inv = warp_and_crop_face(
                img, facial5points, crop_size=(self.size, self.size))

            of = of[..., ::-1].copy()  # BGR->RGB
            of_112 = of_112[..., ::-1].copy()  # BGR->RGB

            # detect orig face quality
            fq_o, fea_o = self.eqface.get_face_quality(of_112)
            if fq_o < self.fqa_thres:
                continue

            # enhance the face
            ef = self.enhance_face(of)

            # detect enhanced face quality
            ss = self.size // 256
            ef_112 = cv2.resize(ef[35 * ss:-33 * ss, 32 * ss:-36 * ss],
                                (112, 112))  # crop roi
            fq_e, fea_e = self.eqface.get_face_quality(ef_112)
            dist = squareform(pdist([fea_o, fea_e], 'cosine')).mean()
            if dist > self.id_thres:
                continue

            tmp_mask = self.mask
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(
                tmp_mask, tfm_inv, (width, height), flags=3)

            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = np.clip(tmp_mask - full_mask, 0, 1)
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

        if self.use_sr and img_sr is not None:
            out_img = cv2.convertScaleAbs(img_sr * (1 - full_mask)
                                          + full_img * full_mask)
        else:
            out_img = cv2.convertScaleAbs(img * (1 - full_mask)
                                          + full_img * full_mask)

        return {OutputKeys.OUTPUT_IMG: out_img.astype(np.uint8)}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

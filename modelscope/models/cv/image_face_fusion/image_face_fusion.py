# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict
from typing import Any, Dict

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.face_detection.peppa_pig_face.facer import FaceAna
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .facegan.gan_wrap import GANWrap
from .facelib.align_trans import (get_f5p, get_reference_facial_points,
                                  warp_and_crop_face)
from .network.aei_flow_net import AEI_Net
from .network.bfm import ParametricFaceModel
from .network.facerecon_model import ReconNetWrapper
from .network.model_irse import Backbone
from .network.ops import warp_affine_torch

logger = get_logger()

__all__ = ['ImageFaceFusion']


@MODELS.register_module(
    Tasks.image_face_fusion, module_name=Models.image_face_fusion)
class ImageFaceFusion(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image face fusion model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.num_kp = 17
        self.id_dim = 512

        self.netG = AEI_Net(
            c_id=self.id_dim, num_kp=self.num_kp, device=self.device)
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoints = torch.load(model_path, map_location='cpu')
        model_state = self.convert_state_dict(checkpoints['state_dict'])
        self.netG.load_state_dict(model_state)
        self.netG = self.netG.to(self.device)
        self.netG.eval()

        self.arcface = Backbone([112, 112], 100, 'ir')
        arcface_path = os.path.join(model_dir, 'faceRecog',
                                    'CurricularFace_Backbone.pth')
        self.arcface.load_state_dict(
            torch.load(arcface_path, map_location='cpu'), strict=False)
        self.arcface = self.arcface.to(self.device)
        self.arcface.eval()

        self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        f_3d_path = os.path.join(model_dir, '3dRecon', 'face_3d.pth')
        self.f_3d.load_state_dict(
            torch.load(f_3d_path, map_location='cpu')['net_recon'])
        self.f_3d = self.f_3d.to(self.device)
        self.f_3d.eval()

        bfm_dir = os.path.join(model_dir, 'BFM')
        self.face_model = ParametricFaceModel(bfm_folder=bfm_dir)
        self.face_model.to(self.device)

        face_enhance_path = os.path.join(model_dir, 'faceEnhance',
                                         '350000-Ns256.pt')
        self.ganwrap = GANWrap(
            model_path=face_enhance_path,
            size=256,
            channel_multiplier=1,
            device=self.device)

        self.facer = FaceAna(model_dir)

        logger.info('load facefusion models done')

        self.mask_init = cv2.imread(os.path.join(model_dir, 'alpha.jpg'))
        self.mask_init = cv2.resize(self.mask_init, (256, 256))
        self.mask = self.image_transform(self.mask_init, is_norm=False)

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        logger.info('init done')

    def convert_state_dict(self, state_dict):
        if not next(iter(state_dict)).startswith('module.'):
            return state_dict
        new_state_dict = OrderedDict()

        split_index = 0
        for cur_key, cur_value in state_dict.items():
            if cur_key.startswith('module.model'):
                split_index = 13
            elif cur_key.startswith('module'):
                split_index = 7

            break

        for k, v in state_dict.items():
            name = k[split_index:]
            new_state_dict[name] = v
        return new_state_dict

    def image_transform(self,
                        image,
                        is_norm=True,
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)):
        image = image.astype(np.float32)
        image = image / 255.0
        if is_norm:
            image -= mean
            image /= std

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.to(self.device)
        return image

    def extract_id(self, np_source, f5p):
        Xs = warp_and_crop_face(
            np_source,
            f5p,
            reference_pts=get_reference_facial_points(default_square=True),
            crop_size=(256, 256))

        Xs = Image.fromarray(Xs)
        Xs = self.test_transform(Xs)
        Xs = Xs.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embeds, Xs_feats = self.arcface(
                F.interpolate(
                    Xs, (112, 112), mode='bilinear', align_corners=True))
        return embeds, Xs

    def detect_face(self, img):
        src_h, src_w, _ = img.shape
        boxes, landmarks, _ = self.facer.run(img)
        if boxes.shape[0] == 0:
            return None
        elif boxes.shape[0] > 1:
            max_area = 0
            max_index = 0
            for i in range(boxes.shape[0]):
                bbox_width = boxes[i][2] - boxes[i][0]
                bbox_height = boxes[i][3] - boxes[i][1]
                area = int(bbox_width) * int(bbox_height)
                if area > max_area:
                    max_index = i
                    max_area = area
            return landmarks[max_index]
        else:
            return landmarks[0]

    def compute_3d_params(self, Xs, Xt):
        kp_fuse = {}
        kp_t = {}

        c_s = self.f_3d(
            F.interpolate(Xs * 0.5 + 0.5, size=224, mode='bilinear'))
        c_t = self.f_3d(
            F.interpolate(Xt * 0.5 + 0.5, size=224, mode='bilinear'))
        c_fuse = torch.cat(((c_s[:, :80] + c_t[:, :80]) / 2, c_t[:, 80:]),
                           dim=1)
        _, _, _, q_fuse = self.face_model.compute_for_render(c_fuse)
        q_fuse = q_fuse / 224
        q_fuse[..., 1] = 1 - q_fuse[..., 1]
        q_fuse = q_fuse * 2 - 1
        delta = int((17 - self.num_kp) / 2)

        _, _, _, q_t = self.face_model.compute_for_render(c_t)
        q_t = q_t / 224
        q_t[..., 1] = 1 - q_t[..., 1]
        q_t = q_t * 2 - 1

        kp_fuse['value'] = q_fuse[:, delta:17 - delta, :]
        kp_t['value'] = q_t[:, delta:17 - delta, :]

        return kp_fuse, kp_t

    def inference(self, template_img, user_img):
        ori_h, ori_w, _ = template_img.shape

        template_img = template_img.cpu().numpy()
        user_img = user_img.cpu().numpy()

        user_img_bgr = user_img[:, :, ::-1]
        landmark_source = self.detect_face(user_img)
        if landmark_source is None:
            logger.warning('No face detected in user image!')
            return template_img
        f5p_user = get_f5p(landmark_source, user_img_bgr)

        template_img_bgr = template_img[:, :, ::-1]
        landmark_template = self.detect_face(template_img)
        if landmark_template is None:
            logger.warning('No face detected in template image!')
            return template_img
        f5p_template = get_f5p(landmark_template, template_img_bgr)

        Xs_embeds, Xs = self.extract_id(user_img, f5p_user)
        Xt, trans_inv = warp_and_crop_face(
            template_img,
            f5p_template,
            reference_pts=get_reference_facial_points(default_square=True),
            crop_size=(256, 256),
            return_trans_inv=True)

        trans_inv = trans_inv.astype(np.float32)
        trans_inv = torch.from_numpy(trans_inv)
        trans_inv = trans_inv.to(self.device)
        Xt_raw = self.image_transform(template_img, is_norm=False)
        Xt = self.image_transform(Xt)

        with torch.no_grad():
            kp_fuse, kp_t = self.compute_3d_params(Xs, Xt)
            Yt, _, _ = self.netG(Xt, Xs_embeds, kp_fuse, kp_t)
            Yt = self.ganwrap.process_tensor(Yt)
            Yt = Yt * 0.5 + 0.5
            Yt = torch.clamp(Yt, 0, 1)

            Yt_trans_inv = warp_affine_torch(Yt, trans_inv, (ori_h, ori_w))
            mask_ = warp_affine_torch(self.mask, trans_inv, (ori_h, ori_w))

            Yt_trans_inv = mask_ * Yt_trans_inv + (1 - mask_) * Xt_raw
            Yt_trans_inv = Yt_trans_inv.squeeze().permute(1, 2,
                                                          0).cpu().numpy()
            Yt_trans_inv = Yt_trans_inv.astype(np.float32)
            out_img = Yt_trans_inv[:, :, ::-1] * 255.

        logger.info('model inference done')

        return out_img.astype(np.uint8)

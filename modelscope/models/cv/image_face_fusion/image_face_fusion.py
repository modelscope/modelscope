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
from .facegan.face_gan import GPEN
from .facelib.align_trans import (get_f5p, get_reference_facial_points,
                                  warp_and_crop_face,
                                  warp_and_crop_face_enhance)
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

        self.facer = FaceAna(model_dir)

        logger.info('load facefusion models done')

        self.mask_init = cv2.imread(os.path.join(model_dir, 'alpha.jpg'))
        self.mask_init = cv2.resize(self.mask_init, (256, 256))
        self.mask = self.image_transform(self.mask_init, is_norm=False)

        face_enhance_path = os.path.join(model_dir, 'faceEnhance',
                                         'GPEN-BFR-1024.pth')

        if not os.path.exists(face_enhance_path):
            logger.warning(
                'model path not found, please update the latest model!')

        self.ganwrap_1024 = GPEN(face_enhance_path, 1024, 2, self.device)

        self.mask_enhance = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask_enhance, (26, 26), (486, 486), (1, 1, 1), -1,
                      cv2.LINE_AA)
        self.mask_enhance = cv2.GaussianBlur(self.mask_enhance, (101, 101), 11)
        self.mask_enhance = cv2.GaussianBlur(self.mask_enhance, (101, 101), 11)

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts_1024 = get_reference_facial_points(
            (1024, 1024), inner_padding_factor, outer_padding, default_square)

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
            return None, None, None
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

            fw = boxes[max_index][2] - boxes[max_index][0]
            fh = boxes[max_index][3] - boxes[max_index][1]
            return landmarks[max_index], fw, fh
        else:
            fw = boxes[0][2] - boxes[0][0]
            fh = boxes[0][3] - boxes[0][1]
            return landmarks[0], fw, fh

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

    def process_enhance(self, im, f5p, fh, fw):
        height, width, _ = im.shape

        of, tfm_inv = warp_and_crop_face_enhance(
            im,
            f5p,
            reference_pts=self.reference_5pts_1024,
            crop_size=(1024, 1024))
        ef, pred = self.ganwrap_1024.process(of)

        tmp_mask = self.mask_enhance
        tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
        tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(im.shape, dtype=np.uint8)

        if min(fh, fw) < 40:
            ef = cv2.pyrDown(ef)
            ef = cv2.pyrDown(ef)
            ef = cv2.pyrUp(ef)
            ef = cv2.pyrUp(ef)
        elif min(fh, fw) < 60:
            ef = cv2.pyrDown(ef)
            ef = cv2.resize(ef, (0, 0), fx=2, fy=2)
            ef = cv2.resize(ef, (0, 0), fx=0.5, fy=0.5)
            ef = cv2.pyrUp(ef)
        elif min(fh, fw) < 80:
            ef = cv2.pyrDown(ef)
            ef = cv2.pyrUp(ef)
        elif min(fh, fw) < 100:
            ef = cv2.pyrDown(ef)
            ef = cv2.resize(ef, (0, 0), fx=2, fy=2)

        tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

        mask = tmp_mask - full_mask
        full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
        full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

        full_mask = full_mask[:, :, np.newaxis]
        im = cv2.convertScaleAbs(im * (1 - full_mask) + full_img * full_mask)
        im = cv2.resize(im, (width, height))
        return im

    def inference(self, template_img, user_img):
        ori_h, ori_w, _ = template_img.shape

        template_img = template_img.cpu().numpy()
        user_img = user_img.cpu().numpy()

        user_img_bgr = user_img[:, :, ::-1]
        landmark_source, _, _ = self.detect_face(user_img)
        if landmark_source is None:
            logger.warning('No face detected in user image!')
            return template_img
        f5p_user = get_f5p(landmark_source, user_img_bgr)

        template_img_bgr = template_img[:, :, ::-1]
        landmark_template, fw, fh = self.detect_face(template_img)
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
            Yt = Yt * 0.5 + 0.5
            Yt = torch.clamp(Yt, 0, 1)

            Yt_trans_inv = warp_affine_torch(Yt, trans_inv, (ori_h, ori_w))
            mask_ = warp_affine_torch(self.mask, trans_inv, (ori_h, ori_w))

            Yt_trans_inv = mask_ * Yt_trans_inv + (1 - mask_) * Xt_raw
            Yt_trans_inv = Yt_trans_inv.squeeze().permute(1, 2,
                                                          0).cpu().numpy()
            Yt_trans_inv = Yt_trans_inv.astype(np.float32)
            out_img = Yt_trans_inv[:, :, ::-1] * 255.
            out_img = self.process_enhance(out_img, f5p_template, fh, fw)

        logger.info('model inference done')

        return out_img.astype(np.uint8)

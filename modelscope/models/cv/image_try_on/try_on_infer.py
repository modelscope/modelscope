# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.nn import functional as F

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .generator import VTONGenerator
from .landmark import VTONLandmark
from .warping import Warping

logger = get_logger()


def load_checkpoint(model, checkpoint_path, device):
    params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(params, strict=False)
    model.to(device)
    model.eval()
    return model


@MODELS.register_module(Tasks.image_try_on, module_name=Models.image_try_on)
class SALForImageTryOn(TorchModel):
    """initialize the image try on model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
    """

    def __init__(self, model_dir, device_id=0, *args, **kwargs):

        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)

        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')

        self.model = VTONGenerator(12, 3, 5, ngf=96, norm_layer=nn.BatchNorm2d)
        self.model = load_checkpoint(
            self.model, model_dir + '/' + ModelFile.TORCH_MODEL_BIN_FILE,
            self.device)

    def forward(self, x, y):
        pred_result = self.model(x, y)
        return pred_result


def infer(ourgen_model, model_path, person_img, garment_img, mask_img, device):

    ourwarp_model = Warping()
    landmark_model = VTONLandmark()
    ourwarp_model = load_checkpoint(ourwarp_model, model_path + '/warp.pth',
                                    device)
    landmark_model.load_state_dict(
        torch.load(model_path + '/landmark.pth', map_location=device))
    landmark_model.to(device).eval()
    input_scale = 4
    with torch.no_grad():
        garment_img = cv2.imread(garment_img)
        garment_img = cv2.cvtColor(garment_img, cv2.COLOR_BGR2RGB)
        clothes = cv2.resize(garment_img, (768, 1024))

        mask_img = cv2.imread(mask_img)
        person_img = cv2.imread(person_img)
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        cm = mask_img[:, :, 0]
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        clothes = input_transform(clothes).unsqueeze(0).to(device)

        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)
        cm = cm.unsqueeze(0).unsqueeze(0)
        cm = torch.FloatTensor((cm.numpy() > 0.5).astype(float)).to(device)

        im = person_img
        h_ori, w_ori = im.shape[0:2]
        im = cv2.resize(im, (768, 1024))
        im = input_transform(im).unsqueeze(0).to(device)

        h, w = 512, 384
        p_down = F.interpolate(im, size=(h, w), mode='bilinear')
        c_down = F.interpolate(clothes, size=(h, w), mode='bilinear')
        c_heatmap, c_property, p_heatmap, p_property = landmark_model(
            c_down, p_down)

        N = c_heatmap.shape[0]
        paired_cloth = clothes[0].cpu()
        color_map = {'1': (0, 0, 255), '0': (255, 0, 0)}
        c_im = (np.array(paired_cloth.permute(1, 2, 0)).copy() + 1) / 2 * 255
        c_im = cv2.cvtColor(c_im, cv2.COLOR_RGB2BGR)
        pred_class = torch.argmax(c_property, dim=1)
        point_ind = torch.argmax(
            c_heatmap.view(N, 32, -1), dim=2).cpu().numpy()
        pred_y, pred_x = 8 * (point_ind // 96), 8 * (point_ind % 96)
        for ind in range(32):
            point_class = int(pred_class[0, ind])
            if point_class < 0.9:
                continue
            point_color = color_map[str(point_class)]
            y, x = pred_y[0][ind], pred_x[0][ind]
            cv2.circle(c_im, (x, y), 2, point_color, 4)
            cv2.putText(
                c_im,
                str(ind), (x + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=point_color,
                thickness=1)
        paired_im = im[0].cpu()
        color_map = {'2': (0, 0, 255), '1': (0, 255, 0), '0': (255, 0, 0)}
        p_im = (np.array(paired_im.permute(1, 2, 0)).copy() + 1) / 2 * 255
        p_im = cv2.cvtColor(p_im, cv2.COLOR_RGB2BGR)
        pred_class = torch.argmax(p_property, dim=1)
        point_ind = torch.argmax(
            p_heatmap.view(N, 32, -1), dim=2).cpu().numpy()
        pred_y, pred_x = 8 * (point_ind // 96), 8 * (point_ind % 96)
        for ind in range(32):
            point_class = int(pred_class[0, ind])
            if point_class < 0.9:
                continue
            point_color = color_map[str(point_class)]
            y, x = pred_y[0][ind], pred_x[0][ind]
            cv2.circle(p_im, (x, y), 2, point_color, 4)
            cv2.putText(
                p_im,
                str(ind), (x + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=point_color,
                thickness=1)
        valid_c_point = np.zeros((32, 2)).astype(np.float32)
        valid_p_point = np.zeros((32, 2)).astype(np.float32)
        c_point_heatmap = -1 * torch.ones(32, 1024, 768)
        p_point_heatmap = -1 * torch.ones(32, 1024, 768)

        cloth_property, person_property = torch.argmax(
            c_property, dim=1), torch.argmax(
                p_property, dim=1)
        cloth_point_ind = torch.argmax(
            c_heatmap.view(N, 32, -1), dim=2).cpu().numpy()
        cloth_y, cloth_x = 8 * (cloth_point_ind // 96), 8 * (
            cloth_point_ind % 96)
        person_point_ind = torch.argmax(
            p_heatmap.view(N, 32, -1), dim=2).cpu().numpy()
        person_y, person_x = 8 * (person_point_ind // 96), 8 * (
            person_point_ind % 96)
        r = 20
        for k in range(32):
            property_c, property_p = cloth_property[0,
                                                    k], person_property[0,
                                                                        k] - 1
            if property_c > 0.1:
                c_x, c_y = cloth_x[0, k], cloth_y[0, k]
                x_min, y_min, x_max, y_max = max(c_x - r - 1, 0), max(
                    c_y - r - 1, 0), min(c_x + r, 768), min(c_y + r, 1024)
                c_point_heatmap[k, y_min:y_max,
                                x_min:x_max] = torch.tensor(property_c)
                valid_c_point[k, 0], valid_c_point[k, 1] = c_x, c_y
            if property_p > -0.99:
                p_x, p_y = person_x[0, k], person_y[0, k]
                x_min, y_min, x_max, y_max = max(p_x - r - 1, 0), max(
                    p_y - r - 1, 0), min(p_x + r, 768), min(p_y + r, 1024)
                p_point_heatmap[k, y_min:y_max,
                                x_min:x_max] = torch.tensor(property_p)
                if property_p > 0:
                    valid_p_point[k, 0], valid_p_point[k, 1] = p_x, p_y

        c_point_plane = torch.tensor(valid_c_point).unsqueeze(0).to(device)
        p_point_plane = torch.tensor(valid_p_point).unsqueeze(0).to(device)
        c_point_heatmap = c_point_heatmap.unsqueeze(0).to(device)
        p_point_heatmap = p_point_heatmap.unsqueeze(0).to(device)

        if input_scale > 1:
            h, w = 1024 // input_scale, 768 // input_scale
            c_point_plane = c_point_plane // input_scale
            p_point_plane = p_point_plane // input_scale
            c_point_heatmap = F.interpolate(
                c_point_heatmap, size=(h, w), mode='nearest')
            p_point_heatmap = F.interpolate(
                p_point_heatmap, size=(h, w), mode='nearest')

            im_down = F.interpolate(im, size=(h, w), mode='bilinear')
            c_down = F.interpolate(cm * clothes, size=(h, w), mode='bilinear')
            cm_down = F.interpolate(cm, size=(h, w), mode='nearest')

        warping_input = [
            c_down, im_down, c_point_heatmap, p_point_heatmap, c_point_plane,
            p_point_plane, cm_down, cm * clothes, device
        ]
        final_warped_cloth, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, \
            delta_y_all, local_warped_cloth_list, fuse_cloth, globalmap, up_cloth = ourwarp_model(warping_input)

        gen_inputs = torch.cat([im, up_cloth], 1)
        gen_outputs = ourgen_model(gen_inputs, p_point_heatmap)

        combine = torch.cat([gen_outputs[0]], 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (w_ori, h_ori))
    return bgr

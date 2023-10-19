# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import numpy as np


def get_fade_out_mask(length, start_value, end_value, fade_start_ratio,
                      fade_end_ratio):
    fade_start_ind = int(length * fade_start_ratio)
    fade_end_ind = int(length * fade_end_ratio)

    left_part = np.array([start_value] * fade_start_ind)
    fade_part = np.linspace(start_value, end_value,
                            fade_end_ind - fade_start_ind)
    len_right = length - len(left_part) - len(fade_part)
    right_part = np.array([end_value] * len_right)

    fade_out_mask = np.concatenate([left_part, fade_part, right_part], axis=0)
    return fade_out_mask


class TexProcesser():

    def __init__(self, model_root):

        self.tex_size = 4096

        self.bald_tex_bg = cv2.imread(
            os.path.join(model_root,
                         'assets/texture/template_bald_tex_2.jpg')).astype(
                             np.float32)
        self.hair_tex_bg = cv2.imread(
            os.path.join(model_root,
                         'assets/texture/template_withHair_tex.jpg')).astype(
                             np.float32)

        self.hair_mask = cv2.imread(
            os.path.join(model_root,
                         'assets/texture/hair_mask_male.png'))[..., 0].astype(
                             np.float32) / 255.0
        self.hair_mask = cv2.resize(self.hair_mask, (4096, 4096 + 1024))

        front_mask = cv2.imread(
            os.path.join(model_root,
                         'assets/texture/face_mask_singleview.jpg')).astype(
                             np.float32) / 255
        front_mask = cv2.resize(front_mask, (1024, 1024))
        front_mask = cv2.resize(front_mask, (0, 0), fx=0.1, fy=0.1)
        front_mask = cv2.erode(front_mask,
                               np.ones(shape=(7, 7), dtype=np.float32))
        front_mask = cv2.GaussianBlur(front_mask, (13, 13), 0)
        self.front_mask = cv2.resize(front_mask,
                                     (self.tex_size, self.tex_size))
        self.binary_front_mask = self.front_mask.copy()
        self.binary_front_mask[(self.front_mask < 0.3)
                               + (self.front_mask > 0.7)] = 0
        self.binary_front_mask[self.binary_front_mask != 0] = 1.0
        self.binary_front_mask_ = self.binary_front_mask.copy()
        self.binary_front_mask_[:int(4096 * 375 / 950)] = 0
        self.binary_front_mask_[int(4096 * 600 / 950):] = 0
        self.binary_front_mask = np.zeros((4096 + 1024, 4096, 3),
                                          dtype=np.float32)
        self.binary_front_mask[:4096, :] = self.binary_front_mask_
        self.front_mask_ = self.front_mask.copy()
        self.front_mask = np.zeros((4096 + 1024, 4096, 3), dtype=np.float32)
        self.front_mask[:4096, :] = self.front_mask_

        self.fg_mask = cv2.imread(
            os.path.join(model_root,
                         'assets/texture/fg_mask.png'))[..., 0].astype(
                             np.float32) / 255.0
        self.fg_mask = cv2.resize(self.fg_mask, (256, 256))
        self.fg_mask = cv2.dilate(self.fg_mask,
                                  np.ones(shape=(13, 13), dtype=np.float32))
        self.fg_mask = cv2.blur(self.fg_mask, (27, 27), 0)
        self.fg_mask = cv2.resize(self.fg_mask, (4096, 4096 + 1024))
        self.fg_mask = self.fg_mask[..., None]

        self.cheek_mask = cv2.imread(
            os.path.join(model_root,
                         'assets/texture/cheek_area_mask.png'))[..., 0].astype(
                             np.float32) / 255.0
        self.cheek_mask = cv2.resize(self.cheek_mask, (4096, 4096 + 1024))
        self.cheek_mask = self.cheek_mask[..., None]

        self.bald_tex_bg = self.bald_tex_bg[:4096]
        self.hair_tex_bg = self.hair_tex_bg[:4096]
        self.fg_mask = self.fg_mask[:4096]
        self.hair_mask = self.hair_mask[:4096]
        self.front_mask = self.front_mask[:4096]
        self.binary_front_mask = self.binary_front_mask[:4096]
        self.front_mask_ = self.front_mask_[:4096]

        self.cheek_mask_left = self.cheek_mask[:4096]
        self.cheek_mask_right = self.cheek_mask[:4096].copy()[:, ::-1]

    def post_process_texture(self, tex_map, hair_tex=True):
        tex_map = cv2.resize(tex_map, (self.tex_size, self.tex_size))

        # if hair_tex is true and there is a dark side, use the mirror texture
        if hair_tex:
            left_cheek_light_mean = np.mean(
                tex_map[self.cheek_mask_left[..., 0] == 1.0])
            right_cheek_light_mean = np.mean(
                tex_map[self.cheek_mask_right[..., 0] == 1.0])

            tex_map_flip = tex_map[:, ::-1, :]
            w = tex_map.shape[1]
            half_w = w // 2
            if left_cheek_light_mean > right_cheek_light_mean * 1.5:
                tex_map[:, half_w:, :] = tex_map_flip[:, half_w:, :]
            elif right_cheek_light_mean > left_cheek_light_mean * 2:
                tex_map[:, :half_w, :] = tex_map_flip[:, :half_w, :]

        # change the color of template texture
        bg_mean_rgb = np.mean(
            self.bald_tex_bg[self.binary_front_mask[..., 0] == 1.0],
            axis=0)[None, None]
        pred_tex_mean_rgb = np.mean(
            tex_map[self.binary_front_mask[..., 0] == 1.0], axis=0)[None,
                                                                    None] * 1.1
        _bald_tex_bg = self.bald_tex_bg.copy()
        _bald_tex_bg = self.bald_tex_bg + (pred_tex_mean_rgb - bg_mean_rgb)

        if hair_tex:
            # inpaint hair
            tex_gray = cv2.cvtColor(
                tex_map.astype(np.uint8),
                cv2.COLOR_BGR2GRAY).astype(np.float32)
            hair_mask = (self.hair_mask == 1.0) * (tex_gray < 120)
            hair_bgr = np.mean(tex_map[hair_mask, :], axis=0) * 0.5
            if hair_bgr is None:
                hair_bgr = 20.0
            _bald_tex_bg[self.hair_mask == 1.0] = hair_bgr

            # fuse
            tex_map = _bald_tex_bg * (1.
                                      - self.fg_mask) + tex_map * self.fg_mask
        else:
            # fuse
            tex_map = _bald_tex_bg * (
                1. - self.front_mask) + tex_map * self.front_mask

        return tex_map

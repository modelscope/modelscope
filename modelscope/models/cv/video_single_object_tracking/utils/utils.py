# The implementation is adopted from OSTrack,
# made publicly available under the MIT License at https://github.com/botaoye/OSTrack/
import math
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def hann1d(sz: int, centered=True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos(
            (2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos(
        (2 * math.pi / (sz + 2)) * torch.arange(0, sz // 2 + 1).float()))
    return torch.cat([w, w[1:sz - sz // 2].flip((0, ))])


def hann2d(sz: torch.Tensor, centered=True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(
        sz[1].item(), centered).reshape(1, 1, 1, -1)


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask


class Preprocessor(object):

    def __init__(self, device: str):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
        if 'cuda' == self.device.type:
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        if 'cuda' == self.device.type:
            img_tensor = torch.tensor(img_arr).to(self.device).float().permute(
                (2, 0, 1)).unsqueeze(dim=0)
        else:
            img_tensor = torch.tensor(img_arr).float().permute(
                (2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = (
            (img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)

        # Deal with the attention mask
        if 'cuda' == self.device.type:
            amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).to(
                self.device).unsqueeze(dim=0)  # (1,H,W)
        else:
            amask_tensor = torch.from_numpy(amask_arr).to(
                torch.bool).unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    if isinstance(x1, torch.Tensor):
        x1 = x1.item()
        y1 = y1.item()
        w = w.item()
        h = h.item()
    return [x1, y1, w, h]


def generate_mask_cond(cfg, bs, device, gt_bbox):
    template_size = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.BACKBONE.STRIDE
    template_feat_size = template_size // stride

    if cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT':
        if template_feat_size == 8:
            index = slice(3, 4)
        elif template_feat_size == 12:
            index = slice(5, 6)
        elif template_feat_size == 7:
            index = slice(3, 4)
        elif template_feat_size == 14:
            index = slice(6, 7)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size],
                                 device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    else:
        raise NotImplementedError

    return box_mask_z


def sample_target(im,
                  target_bb,
                  search_area_factor,
                  output_sz=None,
                  mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad,
                                        x2_pad, cv2.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(
            mask_crop,
            pad=(x1_pad, x2_pad, y1_pad, y2_pad),
            mode='constant',
            value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask,
                              (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
            F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz),
                          mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor,
                            box_extract: torch.Tensor,
                            resize_factor: float,
                            crop_sz: torch.Tensor,
                            normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center
                                          - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def check_box(box: list, image_height, image_width) -> bool:
    """ To check whether the box is within the image range or not
    args:
        box - the bounding box in the form of [x1, y1, x2, y2]
        image_height - the height of the image
        image_width - the width of the image

    returns:
        bool - if box is valid, return True. Otherwise, return False
    """
    assert len(box) == 4, 'box must be in the form of: [x1, y1, x2, y2]'
    if box[0] < 0 or box[0] >= image_width:
        return False
    if box[2] < 0 or box[2] >= image_width:
        return False
    if box[1] < 0 or box[1] >= image_height:
        return False
    if box[3] < 0 or box[3] >= image_height:
        return False
    return True


def timestamp_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time = '%02d:%02d:%06.3f' % (h, m, s)
    return time

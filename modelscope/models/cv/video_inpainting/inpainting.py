""" VideoInpaintingProcess
The implementation here is modified based on STTN,
originally Apache 2.0 License and publicly available at https://github.com/researchmm/STTN
"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

torch.backends.cudnn.enabled = False

w, h = 192, 96
ref_length = 300
neighbor_stride = 20
default_fps = 24
MAX_frame = 300


def video_process(video_input_path):
    video_input = cv2.VideoCapture(video_input_path)
    success, frame = video_input.read()
    if success is False:
        decode_error = 'decode_error'
        w, h, fps = 0, 0, 0
    else:
        decode_error = None
        h, w = frame.shape[0:2]
        fps = video_input.get(cv2.CAP_PROP_FPS)
    video_input.release()

    return decode_error, fps, w, h


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f'Image mode {mode}')


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


_to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])


def get_crop_mask_v1(mask):
    orig_h, orig_w, _ = mask.shape
    if (mask == 255).all():
        return mask, (0, int(orig_h), 0,
                      int(orig_w)), [0, int(orig_h), 0,
                                     int(orig_w)
                                     ], [0, int(orig_h), 0,
                                         int(orig_w)]

    hs = np.min(np.where(mask == 0)[0])
    he = np.max(np.where(mask == 0)[0])
    ws = np.min(np.where(mask == 0)[1])
    we = np.max(np.where(mask == 0)[1])
    crop_box = [ws, hs, we, he]

    mask_h = round(int(orig_h / 2) / 4) * 4
    mask_w = round(int(orig_w / 2) / 4) * 4

    if (hs < mask_h) and (he < mask_h) and (ws < mask_w) and (we < mask_w):
        crop_mask = mask[:mask_h, :mask_w, :]
        res_pix = (0, mask_h, 0, mask_w)
    elif (hs < mask_h) and (he < mask_h) and (ws > mask_w) and (we > mask_w):
        crop_mask = mask[:mask_h, orig_w - mask_w:orig_w, :]
        res_pix = (0, mask_h, orig_w - mask_w, int(orig_w))
    elif (hs > mask_h) and (he > mask_h) and (ws < mask_w) and (we < mask_w):
        crop_mask = mask[orig_h - mask_h:orig_h, :mask_w, :]
        res_pix = (orig_h - mask_h, int(orig_h), 0, mask_w)
    elif (hs > mask_h) and (he > mask_h) and (ws > mask_w) and (we > mask_w):
        crop_mask = mask[orig_h - mask_h:orig_h, orig_w - mask_w:orig_w, :]
        res_pix = (orig_h - mask_h, int(orig_h), orig_w - mask_w, int(orig_w))

    elif (hs < mask_h) and (he < mask_h) and (ws < mask_w) and (we > mask_w):
        crop_mask = mask[:mask_h, :, :]
        res_pix = (0, mask_h, 0, int(orig_w))
    elif (hs < mask_h) and (he > mask_h) and (ws < mask_w) and (we < mask_w):
        crop_mask = mask[:, :mask_w, :]
        res_pix = (0, int(orig_h), 0, mask_w)
    elif (hs > mask_h) and (he > mask_h) and (ws < mask_w) and (we > mask_w):
        crop_mask = mask[orig_h - mask_h:orig_h, :, :]
        res_pix = (orig_h - mask_h, int(orig_h), 0, int(orig_w))
    elif (hs < mask_h) and (he > mask_h) and (ws > mask_w) and (we > mask_w):
        crop_mask = mask[:, orig_w - mask_w:orig_w, :]
        res_pix = (0, int(orig_h), orig_w - mask_w, int(orig_w))
    else:
        crop_mask = mask
        res_pix = (0, int(orig_h), 0, int(orig_w))
    a = ws - res_pix[2]
    b = hs - res_pix[0]
    c = we - res_pix[2]
    d = he - res_pix[0]
    return crop_mask, res_pix, crop_box, [a, b, c, d]


def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def read_mask_oneImage(mpath):
    masks = []
    print('mask_path: {}'.format(mpath))
    start = int(mpath.split('/')[-1].split('mask_')[1].split('_')[0])
    end = int(
        mpath.split('/')[-1].split('mask_')[1].split('_')[1].split('.')[0])
    m = Image.open(mpath)
    m = np.array(m.convert('L'))
    m = np.array(m > 0).astype(np.uint8)
    m = 1 - m
    for i in range(start - 1, end + 1):
        masks.append(Image.fromarray(m * 255))
    return masks


def check_size(h, w):
    is_resize = False
    if h != 240:
        h = 240
        is_resize = True
    if w != 432:
        w = 432
        is_resize = True
    return is_resize


def get_mask_list(mask_path):
    mask_names = os.listdir(mask_path)
    mask_names.sort()

    abs_mask_path = []
    mask_list = []
    begin_list = []
    end_list = []

    for mask_name in mask_names:
        mask_name_tmp = mask_name.split('mask_')[1]
        begin_list.append(int(mask_name_tmp.split('_')[0]))
        end_list.append(int(mask_name_tmp.split('_')[1].split('.')[0]))
        abs_mask_path.append(os.path.join(mask_path, mask_name))
        mask = cv2.imread(os.path.join(mask_path, mask_name))
        mask_list.append(mask)
    return mask_list, begin_list, end_list, abs_mask_path


def inpainting_by_model_balance(model, video_inputPath, mask_path,
                                video_savePath, fps, w_ori, h_ori):

    video_ori = cv2.VideoCapture(video_inputPath)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_save = cv2.VideoWriter(video_savePath, fourcc, fps, (w_ori, h_ori))

    mask_list, begin_list, end_list, abs_mask_path = get_mask_list(mask_path)

    img_npy = []

    for index, mask in enumerate(mask_list):

        masks = read_mask_oneImage(abs_mask_path[index])

        mask, res_pix, crop_for_oriimg, crop_for_inpimg = get_crop_mask_v1(
            mask)
        mask_h, mask_w = mask.shape[0:2]
        is_resize = check_size(mask.shape[0], mask.shape[1])

        begin = begin_list[index]
        end = end_list[index]
        print('begin: {}'.format(begin))
        print('end: {}'.format(end))

        for i in range(begin, end + 1, MAX_frame):
            begin_time = time.time()
            if i + MAX_frame <= end:
                video_length = MAX_frame
            else:
                video_length = end - i + 1

            for frame_count in range(video_length):
                _, frame = video_ori.read()
                img_npy.append(frame)
            frames_temp = []
            for f in img_npy:
                f = Image.fromarray(f)
                i_temp = f.crop(
                    (res_pix[2], res_pix[0], res_pix[3], res_pix[1]))
                a = i_temp.resize((w, h), Image.NEAREST)
                frames_temp.append(a)
            feats_temp = _to_tensors(frames_temp).unsqueeze(0) * 2 - 1
            frames_temp = [np.array(f).astype(np.uint8) for f in frames_temp]
            masks_temp = []
            for m in masks[i - begin:i + video_length - begin]:

                m_temp = m.crop(
                    (res_pix[2], res_pix[0], res_pix[3], res_pix[1]))
                b = m_temp.resize((w, h), Image.NEAREST)
                masks_temp.append(b)
            binary_masks_temp = [
                np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                for m in masks_temp
            ]
            masks_temp = _to_tensors(masks_temp).unsqueeze(0)
            if torch.cuda.is_available():
                feats_temp, masks_temp = feats_temp.cuda(), masks_temp.cuda()
            comp_frames = [None] * video_length
            model.eval()
            with torch.no_grad():
                feats_out = feats_temp * (1 - masks_temp).float()
                feats_out = feats_out.view(video_length, 3, h, w)
                feats_out = model.model.encoder(feats_out)
                _, c, feat_h, feat_w = feats_out.size()
                feats_out = feats_out.view(1, video_length, c, feat_h, feat_w)

            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [
                    i for i in range(
                        max(0, f - neighbor_stride),
                        min(video_length, f + neighbor_stride + 1))
                ]
                ref_ids = get_ref_index(neighbor_ids, video_length)
                with torch.no_grad():
                    pred_feat = model.model.infer(
                        feats_out[0, neighbor_ids + ref_ids, :, :, :],
                        masks_temp[0, neighbor_ids + ref_ids, :, :, :])
                    pred_img = torch.tanh(
                        model.model.decoder(
                            pred_feat[:len(neighbor_ids), :, :, :])).detach()
                    pred_img = (pred_img + 1) / 2
                    pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                    for j in range(len(neighbor_ids)):
                        idx = neighbor_ids[j]
                        img = np.array(pred_img[j]).astype(
                            np.uint8) * binary_masks_temp[idx] + frames_temp[
                                idx] * (1 - binary_masks_temp[idx])
                        if comp_frames[idx] is None:
                            comp_frames[idx] = img
                        else:
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.5 + img.astype(
                                    np.float32) * 0.5
            print('inpainting time:', time.time() - begin_time)
            for f in range(video_length):
                comp = np.array(comp_frames[f]).astype(
                    np.uint8) * binary_masks_temp[f] + frames_temp[f] * (
                        1 - binary_masks_temp[f])
                if is_resize:
                    comp = cv2.resize(comp, (mask_w, mask_h))
                complete_frame = img_npy[f]
                a1, b1, c1, d1 = crop_for_oriimg
                a2, b2, c2, d2 = crop_for_inpimg
                complete_frame[b1:d1, a1:c1] = comp[b2:d2, a2:c2]
                video_save.write(complete_frame)

            img_npy = []

    video_ori.release()

# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/tools/model_surgery.py

import argparse
import os

import torch

COCO_NOVEL_CLASSES = [
    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
]
COCO_BASE_CLASSES = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90
]
COCO_ALL_CLASSES = sorted(COCO_BASE_CLASSES + COCO_NOVEL_CLASSES)
COCO_IDMAP = {v: i for i, v in enumerate(COCO_ALL_CLASSES)}


def surgery(data_type, param_name, is_weight, tar_size, ckpt):
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = ckpt['model'][weight_name]
    prev_cls = pretrained_weight.size(0)
    if 'cls_score' in param_name:
        prev_cls -= 1
    if is_weight:
        feat_size = pretrained_weight.size(1)
        new_weight = torch.rand((tar_size, feat_size))
        torch.nn.init.normal_(new_weight, 0, 0.01)
    else:
        new_weight = torch.zeros(tar_size)
    if data_type == 'coco':
        for idx, c in enumerate(COCO_BASE_CLASSES):
            if 'cls_score' in param_name:
                new_weight[COCO_IDMAP[c]] = pretrained_weight[idx]
            else:
                new_weight[COCO_IDMAP[c] * 4:(COCO_IDMAP[c] + 1) * 4] = \
                    pretrained_weight[idx * 4:(idx + 1) * 4]
    else:
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]
    if 'cls_score' in param_name:
        new_weight[-1] = pretrained_weight[-1]  # bg class
    ckpt['model'][weight_name] = new_weight


def model_surgery(src_path,
                  save_dir,
                  data_type='pascal_voc',
                  method='remove',
                  params_name=[
                      'model.roi_heads.box_predictor.cls_score',
                      'model.roi_heads.box_predictor.bbox_pred'
                  ]):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.
    """

    assert method in ['remove',
                      'randinit'], '{} not implemented'.format(method)

    if data_type == 'coco':
        TAR_SIZE = 80
    elif data_type == 'pascal_voc':
        TAR_SIZE = 20
    else:
        NotImplementedError('{} dataset does not supported'.format(data_type))

    save_name = 'model_reset_' + ('remove' if method == 'remove' else
                                  'surgery') + '.pth'
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    ckpt = torch.load(src_path)
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0

    if method == 'remove':
        for param_name in params_name:
            del ckpt['model'][param_name + '.weight']
            if param_name + '.bias' in ckpt['model']:
                del ckpt['model'][param_name + '.bias']
    elif method == 'randinit':
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name,
                  tar_size) in enumerate(zip(params_name, tar_sizes)):
            surgery(data_type, param_name, True, tar_size, ckpt)
            surgery(data_type, param_name, False, tar_size, ckpt)
    else:
        raise NotImplementedError

    torch.save(ckpt, save_path)

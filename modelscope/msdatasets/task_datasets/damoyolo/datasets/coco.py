# Copyright © Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection

from modelscope.models.cv.tinynas_detection.damo.structures.bounding_box import \
    BoxList

cv2.setNumThreads(0)


class COCODataset(CocoDetection):

    def __init__(self, ann_file, root, transforms=None):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]

        classes = torch.tensor(classes)
        target.add_field('labels', classes)

        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            target.add_field('keypoints', keypoints)

        target = target.clip_to_image(remove_empty=True)

        # PIL to numpy array
        img = np.asarray(img)  # rgb

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def pull_item(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, idx

    def load_anno(self, idx):
        _, anno = super(COCODataset, self).__getitem__(idx)
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]

        return classes

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

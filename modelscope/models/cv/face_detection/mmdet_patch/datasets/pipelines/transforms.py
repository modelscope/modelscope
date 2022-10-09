"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/datasets/pipelines/transforms.py
"""
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy import random


@PIPELINES.register_module()
class RandomSquareCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self,
                 crop_ratio_range=None,
                 crop_choice=None,
                 bbox_clip_border=True):

        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice
        self.bbox_clip_border = bbox_clip_border

        assert (self.crop_ratio_range is None) ^ (self.crop_choice is None)
        if self.crop_ratio_range is not None:
            self.crop_ratio_min, self.crop_ratio_max = self.crop_ratio_range

        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        assert 'gt_bboxes' in results
        boxes = results['gt_bboxes']
        h, w, c = img.shape
        scale_retry = 0
        if self.crop_ratio_range is not None:
            max_scale = self.crop_ratio_max
        else:
            max_scale = np.amax(self.crop_choice)
        while True:
            scale_retry += 1

            if scale_retry == 1 or max_scale > 1.0:
                if self.crop_ratio_range is not None:
                    scale = np.random.uniform(self.crop_ratio_min,
                                              self.crop_ratio_max)
                elif self.crop_choice is not None:
                    scale = np.random.choice(self.crop_choice)
            else:
                scale = scale * 1.2

            for i in range(250):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw

                # TODO +1
                if w == cw:
                    left = 0
                elif w > cw:
                    left = random.randint(0, w - cw)
                else:
                    left = random.randint(w - cw, 0)
                if h == ch:
                    top = 0
                elif h > ch:
                    top = random.randint(0, h - ch)
                else:
                    top = random.randint(h - ch, 0)

                patch = np.array(
                    (int(left), int(top), int(left + cw), int(top + ch)),
                    dtype=np.int)

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                # adjust boxes
                def is_center_of_bboxes_in_patch(boxes, patch):
                    # TODO >=
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = \
                        ((center[:, 0] > patch[0])
                         * (center[:, 1] > patch[1])
                         * (center[:, 0] < patch[2])
                         * (center[:, 1] < patch[3]))
                    return mask

                mask = is_center_of_bboxes_in_patch(boxes, patch)
                if not mask.any():
                    continue
                for key in results.get('bbox_fields', []):
                    boxes = results[key].copy()
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)

                    results[key] = boxes
                    # labels
                    label_key = self.bbox2label.get(key)
                    if label_key in results:
                        results[label_key] = results[label_key][mask]

                    # keypoints field
                    if key == 'gt_bboxes':
                        for kps_key in results.get('keypoints_fields', []):
                            keypointss = results[kps_key].copy()
                            keypointss = keypointss[mask, :, :]
                            if self.bbox_clip_border:
                                keypointss[:, :, :
                                           2] = keypointss[:, :, :2].clip(
                                               max=patch[2:])
                                keypointss[:, :, :
                                           2] = keypointss[:, :, :2].clip(
                                               min=patch[:2])
                            keypointss[:, :, 0] -= patch[0]
                            keypointss[:, :, 1] -= patch[1]
                            results[kps_key] = keypointss

                    # mask fields
                    mask_key = self.bbox2mask.get(key)
                    if mask_key in results:
                        results[mask_key] = results[mask_key][mask.nonzero()
                                                              [0]].crop(patch)

                # adjust the img no matter whether the gt is empty before crop
                rimg = np.ones((ch, cw, 3), dtype=img.dtype) * 128
                patch_from = patch.copy()
                patch_from[0] = max(0, patch_from[0])
                patch_from[1] = max(0, patch_from[1])
                patch_from[2] = min(img.shape[1], patch_from[2])
                patch_from[3] = min(img.shape[0], patch_from[3])
                patch_to = patch.copy()
                patch_to[0] = max(0, patch_to[0] * -1)
                patch_to[1] = max(0, patch_to[1] * -1)
                patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
                patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])
                rimg[patch_to[1]:patch_to[3],
                     patch_to[0]:patch_to[2], :] = img[
                         patch_from[1]:patch_from[3],
                         patch_from[0]:patch_from[2], :]
                img = rimg
                results['img'] = img
                results['img_shape'] = img.shape

                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_iou}, '
        repr_str += f'crop_size={self.crop_size})'
        return repr_str

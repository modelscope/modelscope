"""
The implementation here is modified based on insightface, originally MIT license and publicly available at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/datasets/pipelines/transforms.py
"""
import mmcv
import numpy as np
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from numpy import random


@PIPELINES.register_module()
class ResizeV2(object):
    """Resize images & bbox & mask &kps.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_keypoints(self, results):
        """Resize keypoints with ``results['scale_factor']``."""
        for key in results.get('keypoints_fields', []):
            keypointss = results[key].copy()
            factors = results['scale_factor']
            assert factors[0] == factors[2]
            assert factors[1] == factors[3]
            keypointss[:, :, 0] *= factors[0]
            keypointss[:, :, 1] *= factors[1]
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                keypointss[:, :, 0] = np.clip(keypointss[:, :, 0], 0,
                                              img_shape[1])
                keypointss[:, :, 1] = np.clip(keypointss[:, :, 1], 0,
                                              img_shape[0])
            results[key] = keypointss

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomFlipV2(object):
    """Flip the image & bbox & mask & kps.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image wil
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image wil
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
         of 0.3, vertically with probability of 0.5

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)
        self.count = 0

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def keypoints_flip(self, keypointss, img_shape, direction):
        """Flip keypoints horizontally."""

        assert direction == 'horizontal'
        assert keypointss.shape[-1] == 3
        num_kps = keypointss.shape[1]
        assert num_kps in [4, 5], f'Only Support num_kps=4 or 5, got:{num_kps}'
        assert keypointss.ndim == 3
        flipped = keypointss.copy()
        if num_kps == 5:
            flip_order = [1, 0, 2, 4, 3]
        elif num_kps == 4:
            flip_order = [3, 2, 1, 0]
        for idx, a in enumerate(flip_order):
            flipped[:, idx, :] = keypointss[:, a, :]
        w = img_shape[1]
        flipped[..., 0] = w - flipped[..., 0]
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list)
                                                    - 1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip kps
            for key in results.get('keypoints_fields', []):
                results[key] = self.keypoints_flip(results[key],
                                                   results['img_shape'],
                                                   results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


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
                 bbox_clip_border=True,
                 big_face_ratio=0,
                 big_face_crop_choice=None):

        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice
        self.big_face_crop_choice = big_face_crop_choice
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
        assert big_face_ratio >= 0 and big_face_ratio <= 1.0
        self.big_face_ratio = big_face_ratio

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
        # try augment big face images
        find_bigface = False
        if np.random.random() < self.big_face_ratio:
            min_size = 100  # h and w
            expand_ratio = 0.3  # expand ratio of croped face alongwith both w and h
            bbox = results['gt_bboxes'].copy()
            lmks = results['gt_keypointss'].copy()
            label = results['gt_labels'].copy()
            # filter small faces
            size_mask = ((bbox[:, 2] - bbox[:, 0]) > min_size) * (
                (bbox[:, 3] - bbox[:, 1]) > min_size)
            bbox = bbox[size_mask]
            lmks = lmks[size_mask]
            label = label[size_mask]
            # randomly choose a face that has no overlap with others
            if len(bbox) > 0:
                overlaps = bbox_overlaps(bbox, bbox)
                overlaps -= np.eye(overlaps.shape[0])
                iou_mask = np.sum(overlaps, axis=1) == 0
                bbox = bbox[iou_mask]
                lmks = lmks[iou_mask]
                label = label[iou_mask]
                if len(bbox) > 0:
                    choice = np.random.randint(len(bbox))
                    bbox = bbox[choice]
                    lmks = lmks[choice]
                    label = [label[choice]]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    x1 = bbox[0] - w * expand_ratio
                    x2 = bbox[2] + w * expand_ratio
                    y1 = bbox[1] - h * expand_ratio
                    y2 = bbox[3] + h * expand_ratio
                    x1, x2 = np.clip([x1, x2], 0, img.shape[1])
                    y1, y2 = np.clip([y1, y2], 0, img.shape[0])
                    bbox -= np.tile([x1, y1], 2)
                    lmks -= (x1, y1, 0)

                    find_bigface = True
                    img = img[int(y1):int(y2), int(x1):int(x2), :]
                    results['gt_bboxes'] = np.expand_dims(bbox, axis=0)
                    results['gt_keypointss'] = np.expand_dims(lmks, axis=0)
                    results['gt_labels'] = np.array(label)
                    results['img'] = img

        boxes = results['gt_bboxes']
        h, w, c = img.shape

        if self.crop_ratio_range is not None:
            max_scale = self.crop_ratio_max
        else:
            max_scale = np.amax(self.crop_choice)
        scale_retry = 0
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

            if find_bigface:
                # select a scale from big_face_crop_choice if in big_face mode
                scale = np.random.choice(self.big_face_crop_choice)

            for i in range(250):
                long_side = max(w, h)
                cw = int(scale * long_side)
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
                    dtype=np.int32)

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

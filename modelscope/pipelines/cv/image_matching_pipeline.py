# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_matching, module_name=Pipelines.image_matching)
class ImageMatchingPipeline(Pipeline):
    """ Image Matching Pipeline.

    Example:

    ```python
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks


    task = 'image-matching'
    model_id = 'damo/cv_quadtree_attention_image-matching_outdoor'

    input_location = [
                        ['data/test/images/image_matching1.jpg',
                        'data/test/images/image_matching2.jpg']
                    ]
    estimator = pipeline(Tasks.image_matching, model=self.model_id)
    result = estimator(input_location)
    kpts0, kpts1, conf = result[0][OutputKeys.MATCHES]
    print(f'Found {len(kpts0)} matches')
    ```
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image matching pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Cuda is not available. Image matching model only supports cuda.'
            )

        logger.info('image matching model, pipeline init')

    def resize_image(self, img, max_image_size):
        h, w = img.shape[:2]
        scale = 1
        if max(h, w) > max_image_size:
            scale = max_image_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img, scale

    def compute_paded_size(self, size, div):
        return int(np.ceil(size / div) * div)

    def pad_image(self, img, h=None, w=None, div=32):
        cur_h, cur_w = img.shape[:2]
        if h is None and w is None:
            h, w = cur_h, cur_w
        h_pad, w_pad = self.compute_paded_size(h,
                                               div), self.compute_paded_size(
                                                   w, div)
        img = cv2.copyMakeBorder(
            img,
            0,
            h_pad - cur_h,
            0,
            w_pad - cur_w,
            cv2.BORDER_CONSTANT,
            value=0)
        return img

    def load_image(self, img_name):
        img = LoadImage.convert_to_ndarray(img_name).astype(np.float32)
        img = img / 255.
        # convert rgb to gray
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def preprocess(self, input: Input, max_image_size=1024):
        assert len(input) == 2, 'input should be a list of two images'

        img1 = self.load_image(input[0])
        img1, scale1 = self.resize_image(img1, max_image_size)
        scaled_h1, scaled_w1 = img1.shape[:2]

        img2 = self.load_image(input[1])
        img2, scale2 = self.resize_image(img2, max_image_size)
        scaled_h2, scaled_w2 = img2.shape[:2]

        h_max, w_max = max(scaled_h1, scaled_h2), max(scaled_w1, scaled_w2)
        img1 = self.pad_image(img1, h_max, w_max)
        img2 = self.pad_image(img2, h_max, w_max)

        img1 = torch.from_numpy(img1)[None][None].cuda().float()
        img2 = torch.from_numpy(img2)[None][None].cuda().float()
        return {
            'image0':
            img1,
            'image1':
            img2,
            'preprocess_info':
            [scale1, scale2, scaled_h1, scaled_w1, scaled_h2, scaled_w2]
        }

    def postprocess_match(self, kpt1, kpt2, conf, scale1, scale2, scaled_h1,
                          scaled_w1, scaled_h2, scaled_w2):
        # filter out points outside the image
        valid_match = (kpt1[:, 0] < scaled_w1) & (kpt1[:, 1] < scaled_h1) & (
            kpt2[:, 0] < scaled_w2) & (
                kpt2[:, 1] < scaled_h2)
        kpt1, kpt2 = kpt1[valid_match], kpt2[valid_match]
        kpt1 = kpt1 / scale1
        kpt2 = kpt2 / scale2
        conf = conf[valid_match]

        return kpt1, kpt2, conf

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        matches = results[OutputKeys.MATCHES]

        kpts0 = matches['kpts0'].cpu().numpy()
        kpts1 = matches['kpts1'].cpu().numpy()
        conf = matches['conf'].cpu().numpy()
        preprocess_info = [v.cpu().numpy() for v in inputs['preprocess_info']]
        kpts0, kpts1, conf = self.postprocess_match(kpts0, kpts1, conf,
                                                    *preprocess_info)

        outputs = {
            OutputKeys.MATCHES: [kpts0, kpts1, conf],
        }

        return outputs

    def __call__(self, input, **kwargs):
        """
        Match two images and return the matched keypoints and confidence.

        Args:
            input (`List[List[str]]`): A list of two image paths.

        Return:
            A list of result.
            The list contain the following values:

            - kpts0 -- Matched keypoints in the first image
            - kpts1 -- Matched keypoints in the second image
            - conf -- Confidence of the match
        """
        return super().__call__(input, **kwargs)

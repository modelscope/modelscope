# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from modelscope.outputs.outputs import ModelOutputBase

Tensor = Union['torch.Tensor', 'tf.Tensor']


@dataclass
class DetectionOutput(ModelOutputBase):
    """The output class for object detection models.

    Args:
        class_ids (`Tensor`, *optional*): class id for each object.
        boxes (`Tensor`, *optional*): Bounding box for each detected object in  [left, top, right, bottom] format.
        scores (`Tensor`, *optional*): Detection score for each object.
        keypoints (`Tensor`, *optional*): Keypoints for each object using four corner points in a 8-dim tensor
            in the order of (x, y) for each corner point.

    """

    class_ids: Tensor = None
    scores: Tensor = None
    boxes: Tensor = None
    keypoints: Tensor = None

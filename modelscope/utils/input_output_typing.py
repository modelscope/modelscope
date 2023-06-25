# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List, Tuple, Union

Image = Union[str, 'Image.Image', 'numpy.ndarray']
Text = str
Audio = Union[str, bytes, 'np.ndarray']
Video = Union[str, 'np.ndarray', 'cv2.VideoCapture']

Tensor = Union['torch.Tensor', 'tf.Tensor']

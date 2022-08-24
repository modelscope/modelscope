# Copyright (c) Alibaba, Inc. and its affiliates.
import io
from typing import Any, Dict, Union

import cv2
import numpy as np
import PIL
from numpy import ndarray
from PIL import Image, ImageOps

from modelscope.fileio import File
from modelscope.metainfo import Preprocessors
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS


@PREPROCESSORS.register_module(Fields.cv, Preprocessors.load_image)
class LoadImage:
    """Load an image from file or url.
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        mode (str): See :ref:`PIL.Mode<https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes>`.
    """

    def __init__(self, mode='rgb'):
        self.mode = mode.upper()

    def __call__(self, input: Union[str, Dict[str, str]]):
        """Call functions to load image and get image meta information.
        Args:
            input (str or dict): input image path or input dict with
                a key `filename`.
        Returns:
            dict: The dict contains loaded image.
        """
        if isinstance(input, dict):
            image_path_or_url = input['filename']
        else:
            image_path_or_url = input

        bytes = File.read(image_path_or_url)
        # TODO @wenmeng.zwm add opencv decode as optional
        # we should also look at the input format which is the most commonly
        # used in Mind' image related models
        with io.BytesIO(bytes) as infile:
            img = Image.open(infile)
            img = ImageOps.exif_transpose(img)
            img = img.convert(self.mode)

        results = {
            'filename': image_path_or_url,
            'img': img,
            'img_shape': (img.size[1], img.size[0], 3),
            'img_field': 'img',
        }
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(' f'mode={self.mode})'
        return repr_str

    @staticmethod
    def convert_to_ndarray(input) -> ndarray:
        if isinstance(input, str):
            img = np.array(load_image(input))
        elif isinstance(input, PIL.Image.Image):
            img = np.array(input.convert('RGB'))
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            img = input[:, :, ::-1]
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')
        return img

    @staticmethod
    def convert_to_img(input) -> ndarray:
        if isinstance(input, str):
            img = load_image(input)
        elif isinstance(input, PIL.Image.Image):
            img = input.convert('RGB')
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            img = input[:, :, ::-1]
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')
        return img


def load_image(image_path_or_url: str) -> Image.Image:
    """ simple interface to load an image from file or url

    Args:
        image_path_or_url (str): image file path or http url
    """
    loader = LoadImage()
    return loader(image_path_or_url)['img']


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.image_color_enhance_preprocessor)
class ImageColorEnhanceFinetunePreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)
        self.model_dir: str = model_dir

    @type_assert(object, object)
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (tuple): [sentence1, sentence2]
                sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
                sentence2 (str): a sentence
                    Example:
                        'you are so beautiful.'
        Returns:
            Dict[str, Any]: the preprocessed data
        """

        return data


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.image_denoie_preprocessor)
class ImageDenoisePreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        self.model_dir: str = model_dir

        from .common import Filter

        # TODO: `Filter` should be moved to configurarion file of each model
        self._transforms = [Filter(reserved_keys=['input', 'target'])]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data Dict[str, Any]

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        for t in self._transforms:
            data = t(data)

        return data


@PREPROCESSORS.register_module(
    Fields.cv,
    module_name=Preprocessors.image_portrait_enhancement_preprocessor)
class ImagePortraitEnhancementPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        self.model_dir: str = model_dir

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data Dict[str, Any]

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        return data


@PREPROCESSORS.register_module(
    Fields.cv,
    module_name=Preprocessors.image_instance_segmentation_preprocessor)
class ImageInstanceSegmentationPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        """image instance segmentation preprocessor in the fine-tune scenario
        """

        super().__init__(*args, **kwargs)

        self.training = kwargs.pop('training', True)
        self.preprocessor_train_cfg = kwargs.pop('train', None)
        self.preprocessor_test_cfg = kwargs.pop('val', None)

        self.train_transforms = []
        self.test_transforms = []

        from modelscope.models.cv.image_instance_segmentation.datasets import \
            build_preprocess_transform

        if self.preprocessor_train_cfg is not None:
            if isinstance(self.preprocessor_train_cfg, dict):
                self.preprocessor_train_cfg = [self.preprocessor_train_cfg]
            for cfg in self.preprocessor_train_cfg:
                transform = build_preprocess_transform(cfg)
                self.train_transforms.append(transform)

        if self.preprocessor_test_cfg is not None:
            if isinstance(self.preprocessor_test_cfg, dict):
                self.preprocessor_test_cfg = [self.preprocessor_test_cfg]
            for cfg in self.preprocessor_test_cfg:
                transform = build_preprocess_transform(cfg)
                self.test_transforms.append(transform)

    def train(self):
        self.training = True
        return

    def eval(self):
        self.training = False
        return

    @type_assert(object, object)
    def __call__(self, results: Dict[str, Any]):
        """process the raw input data

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            Dict[str, Any] | None: the preprocessed data
        """

        if self.training:
            transforms = self.train_transforms
        else:
            transforms = self.test_transforms

        for t in transforms:

            results = t(results)

            if results is None:
                return None

        return results

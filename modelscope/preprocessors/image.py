# Copyright (c) Alibaba, Inc. and its affiliates.
import io
from typing import Any, Dict, Union

import torch
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

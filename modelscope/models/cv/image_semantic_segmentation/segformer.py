# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.segmentation import EncoderDecoder

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.image_segmentation, module_name=Models.segformer)
class Segformer(EasyCVBaseModel, EncoderDecoder):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        EncoderDecoder.__init__(self, *args, **kwargs)

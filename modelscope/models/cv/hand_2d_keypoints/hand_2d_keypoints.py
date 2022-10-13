# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.pose import TopDown

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.hand_2d_keypoints, module_name=Models.hand_2d_keypoints)
class Hand2dKeyPoints(EasyCVBaseModel, TopDown):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        TopDown.__init__(self, *args, **kwargs)

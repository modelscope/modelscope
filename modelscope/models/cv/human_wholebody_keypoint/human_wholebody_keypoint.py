# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.pose.top_down import TopDown

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.human_wholebody_keypoint,
    module_name=Models.human_wholebody_keypoint)
class HumanWholeBodyKeypoint(EasyCVBaseModel, TopDown):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        TopDown.__init__(self, *args, **kwargs)

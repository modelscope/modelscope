# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.face.face_keypoint import FaceKeypoint

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.face_2d_keypoints, module_name=Models.face_2d_keypoints)
class Face2DKeypoints(EasyCVBaseModel, FaceKeypoint):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        FaceKeypoint.__init__(self, *args, **kwargs)

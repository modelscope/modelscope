# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Pipelines
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.cv.face_detection_pipeline import \
    FaceDetectionPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.card_detection, module_name=Pipelines.card_detection)
class CardDetectionPipeline(FaceDetectionPipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a card detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        thr = 0.45  # card/face detect use different threshold
        super().__init__(model=model, score_thr=thr, **kwargs)

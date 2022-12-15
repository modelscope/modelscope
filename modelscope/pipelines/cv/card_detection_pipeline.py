# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.typing import Image

logger = get_logger()


@PIPELINES.register_module(
    Tasks.card_detection, module_name=Pipelines.card_detection)
class CardDetectionPipeline(Pipeline):
    """ Card Detection Pipeline.

    Example:

    ```python
    >>> from modelscope.pipelines import pipeline

    >>> detector = pipeline('card-detection', 'damo/cv_resnet_carddetection_scrfd34gkps')
    >>> detector("http://www.modelscope.cn/api/v1/models/damo/cv_resnet_carddetection_scrfd34gkps/repo?Revision=master"
                 "&FilePath=description/card_detection1.jpg")
       {
        "boxes": [
            [
            446.9007568359375,
            36.374977111816406,
            907.0919189453125,
            337.439208984375
            ],
            [
            454.3310241699219,
            336.08477783203125,
            921.26904296875,
            641.7871704101562
            ]
        ],
        "keypoints": [
            [
            457.34710693359375,
            339.02044677734375,
            446.72271728515625,
            52.899078369140625,
            902.8200073242188,
            35.063236236572266,
            908.5877685546875,
            325.62030029296875
            ],
            [
            465.2864074707031,
            642.8411254882812,
            454.38568115234375,
            357.4076232910156,
            902.5343017578125,
            334.18377685546875,
            922.0982055664062,
            621.0704345703125
            ]
        ],
        "scores": [
            0.9296008944511414,
            0.9260380268096924
        ]
        }
    >>> #
    ```
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face detection pipeline for prediction
        Args:
            model: model id on modelscope hub or `ScrfdDetect` Model.
            preprocessor: `SCRFDPreprocessor`.
        """
        super().__init__(model=model, **kwargs)
        assert isinstance(self.model,
                          Model), 'model object is not initialized.'
        detector = self.model.to(self.device)
        self.detector = detector

    def __call__(self, input: Union[Image, List[Image]], **kwargs):
        """
        Detect objects (bounding boxes or keypoints) in the image(s) passed as inputs.

        Args:
            input (`Image` or `List[Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL or opencv directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format.


        Return:
            A dictionary of result or a list of dictionary of result. If the input is an image, a dictionary
            is returned. If input is a list of image, a list of dictionary is returned.

            The dictionary contain the following keys:

            - **scores** (`List[float]`) -- The detection score for each card in the image.
            - **boxes** (`List[float]) -- The bounding boxe [x1, y1, x2, y2] of detected objects in in image's
                original size.
            - **keypoints** (`List[Dict[str, int]]`, optional) -- The corner kepoint [x1, y1, x2, y2, x3, y3, x4, y4]
                of detected object in image's original size.
        """
        return super().__call__(input, **kwargs)

    def preprocess(self, input: Image) -> Dict[str, Any]:
        result = self.preprocessor(input)

        # openmmlab model compatibility
        if 'img_metas' in result:
            from mmcv.parallel import collate, scatter
            result = collate([result], samples_per_gpu=1)
            if next(self.model.parameters()).is_cuda:
                # scatter to specified GPU
                result = scatter(result,
                                 [next(self.model.parameters()).device])[0]
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.detector(**input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

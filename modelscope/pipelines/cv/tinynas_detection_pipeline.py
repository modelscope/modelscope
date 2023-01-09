# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.outputs.cv_outputs import DetectionOutput
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import \
    show_image_object_detection_auto_result
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.domain_specific_object_detection,
    module_name=Pipelines.tinynas_detection)
@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.tinynas_detection)
class TinynasDetectionPipeline(Pipeline):

    def __init__(self,
                 model: str,
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Object detection pipeline, currently only for the tinynas-detection model.

        Args:
            model: A str format model id or model local dir to build the model instance from.
            preprocessor: A preprocessor instance to preprocess the data, if None,
            the pipeline will try to build the preprocessor according to the configuration.json file.
            kwargs: The args needed by the `Pipeline` class.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        return super().preprocess(img)

    def forward(
            self, input: Dict[str,
                              Any]) -> Union[Dict[str, Any], DetectionOutput]:
        """The forward method of this pipeline.

        Args:
            input: The input data output from the `preprocess` procedure.

        Returns:
            A model output, either in a dict format, or in a standard `DetectionOutput` dataclass.
            If outputs a dict, these keys are needed:
                class_ids (`Tensor`, *optional*): class id for each object.
                boxes (`Tensor`, *optional*): Bounding box for each detected object
                    in [left, top, right, bottom] format.
                scores (`Tensor`, *optional*): Detection score for each object.
        """
        return self.model(input['img'])

    def postprocess(
            self, inputs: Union[Dict[str, Any],
                                DetectionOutput]) -> Dict[str, Any]:
        bboxes, scores, labels = inputs['boxes'], inputs['scores'], inputs[
            'class_ids']
        if bboxes is None:
            outputs = {
                OutputKeys.SCORES: [],
                OutputKeys.LABELS: [],
                OutputKeys.BOXES: []
            }
        else:
            outputs = {
                OutputKeys.SCORES: scores,
                OutputKeys.LABELS: labels,
                OutputKeys.BOXES: bboxes
            }
        return outputs

    def show_result(self, img_path, result, save_path=None):
        show_image_object_detection_auto_result(img_path, result, save_path)

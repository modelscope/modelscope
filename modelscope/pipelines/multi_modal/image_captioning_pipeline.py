from typing import Any, Dict, Union

from modelscope.preprocessors import OfaImageCaptionPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ..base import Model, Pipeline
from ..builder import PIPELINES

logger = get_logger()


@PIPELINES.register_module(Tasks.image_captioning, module_name='ofa')
class ImageCaptionPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: [Preprocessor] = None,
                 **kwargs):
        super().__init__()
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or OfaForImageCaptioning'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        if preprocessor is None and pipe_model:
            preprocessor = OfaImageCaptionPreprocessor(model_dir=model)
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

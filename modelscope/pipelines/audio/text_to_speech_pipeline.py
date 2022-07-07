from typing import Any, Dict, List

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor, TextToTacotronSymbols
from modelscope.utils.constant import Tasks

__all__ = ['TextToSpeechSambertHifigan16kPipeline']


@PIPELINES.register_module(
    Tasks.text_to_speech, module_name=Pipelines.sambert_hifigan_16k_tts)
class TextToSpeechSambertHifigan16kPipeline(Pipeline):

    def __init__(self,
                 model: List[str] = None,
                 preprocessor: Preprocessor = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        assert len(model) == 3, 'model number should be 3'
        if preprocessor is None:
            lang_type = 'pinyin'
            if 'lang_type' in kwargs:
                lang_type = kwargs.lang_type
            preprocessor = TextToTacotronSymbols(model[0], lang_type=lang_type)
        models = [model[1], model[2]]
        super().__init__(model=models, preprocessor=preprocessor, **kwargs)
        self._am = self.models[0]
        self._vocoder = self.models[1]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        texts = inputs['texts']
        audio_total = np.empty((0), dtype='int16')
        for line in texts:
            line = line.strip().split('\t')
            audio = self._vocoder.forward(self._am.forward(line[1]))
            audio_total = np.append(audio_total, audio, axis=0)
        return {'output': audio_total}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

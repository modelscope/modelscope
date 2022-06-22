import time
from typing import Any, Dict, List

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.audio.tts.am import SambertNetHifi16k
from modelscope.models.audio.tts.vocoder import Hifigan16k
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TextToTacotronSymbols, build_preprocessor
from modelscope.utils.constant import Fields, Tasks

__all__ = ['TextToSpeechSambertHifigan16kPipeline']


@PIPELINES.register_module(
    Tasks.text_to_speech, module_name=Pipelines.sambert_hifigan_16k_tts)
class TextToSpeechSambertHifigan16kPipeline(Pipeline):

    def __init__(self,
                 config_file: str = None,
                 model: List[Model] = None,
                 preprocessor: TextToTacotronSymbols = None,
                 **kwargs):
        super().__init__(
            config_file=config_file,
            model=model,
            preprocessor=preprocessor,
            **kwargs)
        assert len(model) == 2, 'model number should be 2'
        self._am = model[0]
        self._vocoder = model[1]
        self._preprocessor = preprocessor

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

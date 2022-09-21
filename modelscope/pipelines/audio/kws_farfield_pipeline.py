# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import wave
from typing import Any, Dict

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.keyword_spotting,
    module_name=Pipelines.speech_dfsmn_kws_char_farfield)
class KWSFarfieldPipeline(Pipeline):
    r"""A Keyword Spotting Inference Pipeline .

    When invoke the class with pipeline.__call__(), it accept only one parameter:
        inputs(str): the path of wav file
    """
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 2

    def __init__(self, model, **kwargs):
        """
        use `model` to create a kws far field pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()
        frame_size = self.INPUT_CHANNELS * self.SAMPLE_WIDTH
        self._nframe = self.model.size_in // frame_size
        self.frame_count = 0

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        if isinstance(inputs, bytes):
            return dict(input_file=inputs)
        elif isinstance(inputs, str):
            return dict(input_file=inputs)
        elif isinstance(inputs, Dict):
            return inputs
        else:
            raise ValueError(f'Not supported input type: {type(inputs)}')

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        input_file = inputs['input_file']
        if isinstance(input_file, str):
            input_file = File.read(input_file)
        if isinstance(input_file, bytes):
            input_file = io.BytesIO(input_file)
        self.frame_count = 0
        kws_list = []
        with wave.open(input_file, 'rb') as fin:
            if 'output_file' in inputs:
                with wave.open(inputs['output_file'], 'wb') as fout:
                    fout.setframerate(self.SAMPLE_RATE)
                    fout.setnchannels(self.OUTPUT_CHANNELS)
                    fout.setsampwidth(self.SAMPLE_WIDTH)
                    self._process(fin, kws_list, fout)
            else:
                self._process(fin, kws_list)
        return {OutputKeys.KWS_LIST: kws_list}

    def _process(self,
                 fin: wave.Wave_read,
                 kws_list,
                 fout: wave.Wave_write = None):
        data = fin.readframes(self._nframe)
        while len(data) >= self.model.size_in:
            self.frame_count += self._nframe
            result = self.model.forward_decode(data)
            if fout:
                fout.writeframes(result['pcm'])
            if 'kws' in result:
                result['kws']['offset'] += self.frame_count / self.SAMPLE_RATE
                kws_list.append(result['kws'])
            data = fin.readframes(self._nframe)

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

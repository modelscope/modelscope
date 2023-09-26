# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import Mapping

import numpy as np
import onnx
import torch

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.torch_model_exporter import TorchModelExporter
from modelscope.metainfo import Models
from modelscope.utils.constant import ModelFile, Tasks


@EXPORTERS.register_module(
    Tasks.ocr_recognition, module_name=Models.ocr_recognition)
class OCRRecognitionExporter(TorchModelExporter):

    def export_onnx(self,
                    output_dir: str,
                    opset=11,
                    input_shape=(1, 3, 32, 640)):
        onnx_file = os.path.join(output_dir, ModelFile.ONNX_MODEL_FILE)
        dummy_input = torch.randn(*input_shape)
        self.model.onnx_export = True
        self.model.eval()
        _ = self.model(dummy_input)
        torch.onnx._export(
            self.model,
            dummy_input,
            onnx_file,
            input_names=[
                'images',
            ],
            output_names=[
                'pred',
            ],
            opset_version=opset)
        return {'model', onnx_file}

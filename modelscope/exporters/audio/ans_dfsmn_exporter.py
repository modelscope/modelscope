# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.torch_model_exporter import TorchModelExporter
from modelscope.metainfo import Models
from modelscope.utils.constant import ModelFile, Tasks

INPUT_NAME = 'input'
OUTPUT_NAME = 'output'


@EXPORTERS.register_module(
    Tasks.acoustic_noise_suppression, module_name=Models.speech_dfsmn_ans)
class ANSDFSMNExporter(TorchModelExporter):

    def export_onnx(self, output_dir: str, opset=9, **kwargs):
        """Export the model as onnx format files.

        Args:
            output_dir: The output dir.
            opset: The version of the ONNX operator set to use.
            kwargs:
                device: The device used to forward.
        Returns:
            A dict containing the model key - model file path pairs.
        """
        model = self.model if 'model' not in kwargs else kwargs.pop('model')
        device_name = 'cpu' if 'device' not in kwargs else kwargs.pop('device')
        model_bin_file = os.path.join(model.model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE)
        if os.path.exists(model_bin_file):
            checkpoint = torch.load(model_bin_file, map_location='cpu')
            model.load_state_dict(checkpoint)
        onnx_file = os.path.join(output_dir, ModelFile.ONNX_MODEL_FILE)

        with torch.no_grad():
            model.eval()
            device = torch.device(device_name)
            model.to(device)
            model_script = torch.jit.script(model)
            fbank_input = torch.zeros((1, 3, 120), dtype=torch.float32)
            torch.onnx.export(
                model_script,
                fbank_input,
                onnx_file,
                opset_version=opset,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={
                    INPUT_NAME: {
                        0: 'batch_size',
                        1: 'number_of_frame'
                    },
                    OUTPUT_NAME: {
                        0: 'batch_size',
                        1: 'number_of_frame'
                    }
                })
        return {'model': onnx_file}

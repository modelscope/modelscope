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


def convert_ndarray_to_list(input_dict):
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            input_dict[key] = value.tolist()
        elif isinstance(value, dict):
            convert_ndarray_to_list(value)
    return input_dict


@EXPORTERS.register_module(Tasks.face_detection, module_name=Models.scrfd)
class FaceDetectionSCRFDExporter(TorchModelExporter):

    def export_onnx(self,
                    output_dir: str,
                    opset=9,
                    simplify=True,
                    dynamic=False,
                    **kwargs):
        """Export the model as onnx format files.

        Args:
            output_dir: The output dir.
            opset: The version of the ONNX operator set to use.
            simplify: simplify the onnx model
            dynamic: use dynamic input size

        Returns:
            A dict containing the model key - model file path pairs.
        """
        from mmdet.core.export import preprocess_example_input
        input_shape = (1, 3, 640, 640)
        input_config = {
            'input_shape': input_shape,
            'input_path': 'data/test/images/face_detection2.jpeg',
            'normalize_cfg': {
                'mean': [127.5, 127.5, 127.5],
                'std': [128.0, 128.0, 128.0]
            }
        }

        model = self.model.detector.module if 'model' not in kwargs else kwargs.pop(
            'model')
        model = model.cpu().eval()
        output_file = os.path.join(output_dir, ModelFile.ONNX_MODEL_FILE)
        if simplify or dynamic:
            ori_output_file = output_file.split('.onnx')[0] + '_ori.onnx'
        else:
            ori_output_file = output_file
        one_img, one_meta = preprocess_example_input(input_config)
        tensor_data = [one_img]
        if 'show_img' in one_meta:
            del one_meta['show_img']

        one_meta = convert_ndarray_to_list(one_meta)
        model.forward = partial(
            model.forward, img_metas=[[one_meta]], return_loss=False)
        torch.onnx.export(
            model,
            tensor_data,
            ori_output_file,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=opset)

        if simplify or dynamic:
            model = onnx.load(ori_output_file)
            if dynamic:
                model.graph.input[0].type.tensor_type.shape.dim[
                    2].dim_param = '?'
                model.graph.input[0].type.tensor_type.shape.dim[
                    3].dim_param = '?'
            if simplify:
                from onnxsim import simplify
                if dynamic:
                    input_shapes = {
                        model.graph.input[0].name: list(input_shape)
                    }
                    model, check = simplify(
                        model, overwrite_input_shapes=input_shapes)
                else:
                    model, check = simplify(model)
                assert check, 'Simplified ONNX model could not be validated'
            onnx.save(model, output_file)
            os.remove(ori_output_file)
        print(f'Successfully exported ONNX model: {output_file}')
        return {'model': output_file}

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, Mapping

import torch
from torch import nn
from torch.onnx import export as onnx_export

from modelscope.models import TorchModel
from modelscope.outputs import ModelOutputBase
from modelscope.pipelines.base import collate_fn
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.regress_test_utils import (compare_arguments_nested,
                                                 numpify_tensor_nested)
from .base import Exporter

logger = get_logger()


class TorchModelExporter(Exporter):
    """The torch base class of exporter.

    This class provides the default implementations for exporting onnx and torch script.
    Each specific model may implement its own exporter by overriding the export_onnx/export_torch_script,
    and to provide implementations for generate_dummy_inputs/inputs/outputs methods.
    """

    def export_onnx(self, output_dir: str, opset=13, **kwargs):
        """Export the model as onnx format files.

        In some cases,  several files may be generated,
        So please return a dict which contains the generated name with the file path.

        Args:
            opset: The version of the ONNX operator set to use.
            output_dir: The output dir.
            kwargs:
                model: A model instance which will replace the exporting of self.model.
                In this default implementation,
                you can pass the arguments needed by _torch_export_onnx, other unrecognized args
                will be carried to generate_dummy_inputs as extra arguments (such as input shape).

        Returns:
            A dict containing the model key - model file path pairs.
        """
        model = self.model if 'model' not in kwargs else kwargs.pop('model')
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            model = model.model
        onnx_file = os.path.join(output_dir, ModelFile.ONNX_MODEL_FILE)
        self._torch_export_onnx(model, onnx_file, opset=opset, **kwargs)
        return {'model': onnx_file}

    def export_torch_script(self, output_dir: str, **kwargs):
        """Export the model as torch script files.

        In some cases,  several files may be generated,
        So please return a dict which contains the generated name with the file path.

        Args:
            output_dir: The output dir.
            kwargs:
            model: A model instance which will replace the exporting of self.model.
            In this default implementation,
            you can pass the arguments needed by _torch_export_torch_script, other unrecognized args
            will be carried to generate_dummy_inputs as extra arguments (like input shape).

        Returns:
            A dict contains the model name with the model file path.
        """
        model = self.model if 'model' not in kwargs else kwargs.pop('model')
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            model = model.model
        ts_file = os.path.join(output_dir, ModelFile.TS_MODEL_FILE)
        # generate ts by tracing
        self._torch_export_torch_script(model, ts_file, **kwargs)
        return {'model': ts_file}

    def generate_dummy_inputs(self, **kwargs) -> Dict[str, Any]:
        """Generate dummy inputs for model exportation to onnx or other formats by tracing.

        Returns:
            Dummy inputs.
        """
        return None

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """Return an ordered dict contains the model's input arguments name with their dynamic axis.

        About the information of dynamic axis please check the dynamic_axes argument of torch.onnx.export function
        """
        return None

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """Return an ordered dict contains the model's output arguments name with their dynamic axis.

        About the information of dynamic axis please check the dynamic_axes argument of torch.onnx.export function
        """
        return None

    @staticmethod
    def _decide_input_format(model, args):
        import inspect

        def _signature(model) -> inspect.Signature:
            should_be_callable = getattr(model, 'forward', model)
            if callable(should_be_callable):
                return inspect.signature(should_be_callable)
            raise ValueError('model has no forward method and is not callable')

        try:
            sig = _signature(model)
        except ValueError as e:
            logger.warning('%s, skipping _decide_input_format' % e)
            return args
        try:
            ordered_list_keys = list(sig.parameters.keys())
            if ordered_list_keys[0] == 'self':
                ordered_list_keys = ordered_list_keys[1:]
            args_dict: Dict = {}
            if isinstance(args, list):
                args_list = args
            elif isinstance(args, tuple):
                args_list = list(args)
            else:
                args_list = [args]
            if isinstance(args_list[-1], Mapping):
                args_dict = args_list[-1]
                args_list = args_list[:-1]
            n_nonkeyword = len(args_list)
            for optional_arg in ordered_list_keys[n_nonkeyword:]:
                if optional_arg in args_dict:
                    args_list.append(args_dict[optional_arg])
                # Check if this arg has a default value
                else:
                    param = sig.parameters[optional_arg]
                    if param.default != param.empty:
                        args_list.append(param.default)
            args = args_list if isinstance(args, list) else tuple(args_list)
        # Cases of models with no input args
        except IndexError:
            logger.warning('No input args, skipping _decide_input_format')
        except Exception as e:
            logger.warning('Skipping _decide_input_format\n {}'.format(
                e.args[0]))

        return args

    def _torch_export_onnx(self,
                           model: nn.Module,
                           output: str,
                           opset: int = 13,
                           device: str = 'cpu',
                           validation: bool = True,
                           rtol: float = None,
                           atol: float = None,
                           **kwargs):
        """Export the model to an onnx format file.

        Args:
            model: A torch.nn.Module instance to export.
            output: The output file.
            opset: The version of the ONNX operator set to use.
            device: The device used to forward.
            validation: Whether validate the export file.
            rtol: The rtol used to regress the outputs.
            atol: The atol used to regress the outputs.
            kwargs:
                dummy_inputs: A dummy inputs which will replace the calling of self.generate_dummy_inputs().
                inputs: An inputs structure which will replace the calling of self.inputs.
                outputs: An outputs structure which will replace the calling of self.outputs.
        """

        dummy_inputs = self.generate_dummy_inputs(
            **kwargs) if 'dummy_inputs' not in kwargs else kwargs.pop(
                'dummy_inputs')
        inputs = self.inputs if 'inputs' not in kwargs else kwargs.pop(
            'inputs')
        outputs = self.outputs if 'outputs' not in kwargs else kwargs.pop(
            'outputs')
        if dummy_inputs is None or inputs is None or outputs is None:
            raise NotImplementedError(
                'Model property dummy_inputs,inputs,outputs must be set.')

        with torch.no_grad():
            model.eval()
            device = torch.device(device)
            model.to(device)
            dummy_inputs = collate_fn(dummy_inputs, device)

            if isinstance(dummy_inputs, Mapping):
                dummy_inputs = dict(dummy_inputs)
            onnx_outputs = list(outputs.keys())

            with replace_call():
                onnx_export(
                    model,
                    (dummy_inputs, ),
                    f=output,
                    input_names=list(inputs.keys()),
                    output_names=onnx_outputs,
                    dynamic_axes={
                        name: axes
                        for name, axes in chain(inputs.items(),
                                                outputs.items())
                    },
                    do_constant_folding=True,
                    opset_version=opset,
                )

        if validation:
            try:
                import onnx
                import onnxruntime as ort
            except ImportError:
                logger.warning(
                    'Cannot validate the exported onnx file, because '
                    'the installation of onnx or onnxruntime cannot be found')
                return
            onnx_model = onnx.load(output)
            onnx.checker.check_model(onnx_model)
            ort_session = ort.InferenceSession(output)
            with torch.no_grad():
                model.eval()
                outputs_origin = model.forward(
                    *self._decide_input_format(model, dummy_inputs))
            if isinstance(outputs_origin, (Mapping, ModelOutputBase)):
                outputs_origin = list(
                    numpify_tensor_nested(outputs_origin).values())
            elif isinstance(outputs_origin, (tuple, list)):
                outputs_origin = list(numpify_tensor_nested(outputs_origin))
            outputs = ort_session.run(
                onnx_outputs,
                numpify_tensor_nested(dummy_inputs),
            )
            outputs = numpify_tensor_nested(outputs)
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
            elif isinstance(outputs, tuple):
                outputs = list(outputs)

            tols = {}
            if rtol is not None:
                tols['rtol'] = rtol
            if atol is not None:
                tols['atol'] = atol
            if not compare_arguments_nested('Onnx model output match failed',
                                            outputs, outputs_origin, **tols):
                raise RuntimeError(
                    'export onnx failed because of validation error.')

    def _torch_export_torch_script(self,
                                   model: nn.Module,
                                   output: str,
                                   device: str = 'cpu',
                                   validation: bool = True,
                                   rtol: float = None,
                                   atol: float = None,
                                   strict: bool = True,
                                   **kwargs):
        """Export the model to a torch script file.

        Args:
            model: A torch.nn.Module instance to export.
            output: The output file.
            device: The device used to forward.
            validation: Whether validate the export file.
            rtol: The rtol used to regress the outputs.
            atol: The atol used to regress the outputs.
            strict: strict mode in torch script tracing.
            kwargs:
                dummy_inputs: A dummy inputs which will replace the calling of self.generate_dummy_inputs().
        """

        model.eval()
        dummy_param = 'dummy_inputs' not in kwargs
        dummy_inputs = self.generate_dummy_inputs(
            **kwargs) if dummy_param else kwargs.pop('dummy_inputs')
        if dummy_inputs is None:
            raise NotImplementedError(
                'Model property dummy_inputs must be set.')
        dummy_inputs = collate_fn(dummy_inputs, device)
        if isinstance(dummy_inputs, Mapping):
            dummy_inputs_filter = []
            for _input in self._decide_input_format(model, dummy_inputs):
                if _input is not None:
                    dummy_inputs_filter.append(_input)
                else:
                    break

            if len(dummy_inputs) != len(dummy_inputs_filter):
                logger.warning(
                    f'Dummy inputs is not continuous in the forward method, '
                    f'origin length: {len(dummy_inputs)}, '
                    f'the length after filtering: {len(dummy_inputs_filter)}')
            dummy_inputs = dummy_inputs_filter

        with torch.no_grad():
            model.eval()
            with replace_call():
                traced_model = torch.jit.trace(
                    model, tuple(dummy_inputs), strict=strict)
        torch.jit.save(traced_model, output)

        if validation:
            ts_model = torch.jit.load(output)
            with torch.no_grad():
                model.eval()
                ts_model.eval()
                outputs = ts_model.forward(*dummy_inputs)
                outputs = numpify_tensor_nested(outputs)
                outputs_origin = model.forward(*dummy_inputs)
                outputs_origin = numpify_tensor_nested(outputs_origin)
                if isinstance(outputs, dict):
                    outputs = list(outputs.values())
                if isinstance(outputs_origin, dict):
                    outputs_origin = list(outputs_origin.values())
            tols = {}
            if rtol is not None:
                tols['rtol'] = rtol
            if atol is not None:
                tols['atol'] = atol
            if not compare_arguments_nested(
                    'Torch script model output match failed', outputs,
                    outputs_origin, **tols):
                raise RuntimeError(
                    'export torch script failed because of validation error.')


@contextmanager
def replace_call():
    """This function is used to recover the original call method.

    The Model class of modelscope overrides the call method. When exporting to onnx or torchscript, torch will
    prepare the parameters as the prototype of forward method, and trace the call method, this causes
    problems. Here we recover the call method to the default implementation of torch.nn.Module, and change it
    back after the tracing was done.
    """
    TorchModel.call_origin, TorchModel.__call__ = TorchModel.__call__, TorchModel._call_impl
    yield
    TorchModel.__call__ = TorchModel.call_origin
    del TorchModel.call_origin

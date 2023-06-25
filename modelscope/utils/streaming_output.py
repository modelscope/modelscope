# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Generator, List, Union

import torch

from modelscope.pipelines.base import Input
from modelscope.utils.constant import Frameworks
from modelscope.utils.device import device_placement


class StreamingOutputMixin:

    def stream(self, *args, **kwargs) -> Generator:
        """
        Support the input of Model and Pipeline.
        The output is a `Generator` type,
        which conforms to the output standard of modelscope.
        """
        raise NotImplementedError


class PipelineStreamingOutputMixin(StreamingOutputMixin):

    def stream(self, input: Union[Input, List[Input]], *args,
               **kwargs) -> Generator:
        """
        Similar to the `Pipeline.__call__` method.
        it supports the input that the pipeline can accept,
        and also supports batch input.

        self.model must be a subclass of StreamingOutputMixin
        and implement the stream method.
        """
        assert isinstance(self.model, StreamingOutputMixin
                          ), 'pipeline.model must be StreamingOutputMixin!'
        if (self.model or (self.has_multiple_models and self.models[0])):
            if not self._model_prepare:
                self.prepare_model()

        batch_size = kwargs.pop('batch_size', None)
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            **kwargs)

        if isinstance(input, list):
            model_input_list = [
                self._preprocess_with_check(i, preprocess_params)
                for i in input
            ]

            if batch_size is None:
                output = []
                for ele in model_input_list:
                    output.append(
                        self._stream_single(ele, forward_params,
                                            postprocess_params))
            else:
                output = self._stream_batch(model_input_list, batch_size,
                                            forward_params, postprocess_params)

        else:
            model_input = self._preprocess_with_check(input, preprocess_params)
            output = self._stream_single(model_input, forward_params,
                                         postprocess_params)
        return output

    def _preprocess_with_check(
            self, input: Input,
            preprocess_params: Dict[str, Any]) -> Dict[str, Any]:
        self._check_input(input)
        return self.preprocess(input, **preprocess_params)

    def _stream_single(self, model_input: Dict[str, Any],
                       forward_params: Dict[str, Any],
                       postprocess_params: Dict[str, Any]) -> Generator:

        with device_placement(self.framework, self.device_name):
            if self.framework == Frameworks.torch:
                with torch.no_grad():
                    if self._auto_collate:
                        model_input = self._collate_fn(model_input)
                    stream = self.model.stream(model_input, **forward_params)
            else:
                stream = self.model.stream(model_input, **forward_params)

            for out in stream:
                out = self.postprocess(out, **postprocess_params)
                self._check_output(out)
                yield out

    def _stream_batch(self, model_input_list: List[Dict[str, Any]],
                      batch_size: int, forward_params: Dict[str, Any],
                      postprocess_params: Dict[str, Any]) -> Generator:

        stream_list = []
        real_batch_sizes = []
        with device_placement(self.framework, self.device_name):
            for i in range(0, len(model_input_list), batch_size):
                end = min(i + batch_size, len(model_input_list))
                real_batch_size = end - i
                real_batch_sizes.append(real_batch_size)

                batched_out = self._batch(model_input_list[i:end])
                if self.framework == Frameworks.torch:
                    with torch.no_grad():
                        if self._auto_collate:
                            batched_out = self._collate_fn(batched_out)
                        stream_list.append(
                            self.model.stream(batched_out, **forward_params))
                else:
                    stream_list.append(
                        self.model.stream(batched_out, **forward_params))

            output_list = [None] * len(model_input_list)
            stop_streams = 0
            while stop_streams < len(stream_list):
                stop_streams = 0
                for i, (stream, real_batch_size) in enumerate(
                        zip(stream_list, real_batch_sizes)):
                    try:
                        batched_out = next(stream)
                        for batch_idx in range(real_batch_size):
                            out = {}
                            for k, element in batched_out.items():
                                if element is not None:
                                    if isinstance(element, (tuple, list)):
                                        if isinstance(element[0],
                                                      torch.Tensor):
                                            out[k] = type(element)(
                                                e[batch_idx:batch_idx + 1]
                                                for e in element)
                                        else:
                                            # Compatible with traditional pipelines
                                            out[k] = element[batch_idx]
                                    else:
                                        out[k] = element[batch_idx:batch_idx
                                                         + 1]
                            out = self.postprocess(out, **postprocess_params)
                            self._check_output(out)
                            output_index = i * batch_size + batch_idx
                            output_list[output_index] = out
                    except StopIteration:
                        stop_streams += 1
                yield output_list

        return output_list

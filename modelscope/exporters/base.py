# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import ABC, abstractmethod
from typing import Dict, Union

from modelscope.models import Model
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from .builder import build_exporter

logger = get_logger()


class Exporter(ABC):
    """Exporter base class to output model to onnx, torch_script, graphdef, etc.
    """

    def __init__(self, model=None):
        self.model = model

    @classmethod
    def from_model(cls, model: Union[Model, str], **kwargs):
        """Build the Exporter instance.

        Args:
            model: A Model instance or a model id or a model dir, the configuration.json file besides to which
            will be used to create the exporter instance.
            kwargs: Extra kwargs used to create the Exporter instance.

        Returns:
            The Exporter instance
        """
        if isinstance(model, str):
            model = Model.from_pretrained(model)

        assert hasattr(model, 'model_dir')
        model_dir = model.model_dir
        cfg = Config.from_file(
            os.path.join(model_dir, ModelFile.CONFIGURATION))
        task_name = cfg.task
        if hasattr(model, 'group_key'):
            task_name = model.group_key
        model_cfg = cfg.model
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type
        export_cfg = ConfigDict({'type': model_cfg.type})
        if hasattr(cfg, 'export'):
            export_cfg.update(cfg.export)
        export_cfg['model'] = model
        try:
            exporter = build_exporter(export_cfg, task_name, kwargs)
        except KeyError as e:
            raise KeyError(
                f'The exporting of model \'{model_cfg.type}\' with task: \'{task_name}\' '
                f'is not supported currently.') from e
        return exporter

    @abstractmethod
    def export_onnx(self, output_dir: str, opset=13, **kwargs):
        """Export the model as onnx format files.

        In some cases,  several files may be generated,
        So please return a dict which contains the generated name with the file path.

        Args:
            opset: The version of the ONNX operator set to use.
            output_dir: The output dir.
            kwargs: In this default implementation,
                kwargs will be carried to generate_dummy_inputs as extra arguments (like input shape).

        Returns:
            A dict contains the model name with the model file path.
        """
        pass

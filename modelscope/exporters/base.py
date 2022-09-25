# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import ABC, abstractmethod

from modelscope.models import Model
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModelFile
from .builder import build_exporter


class Exporter(ABC):
    """Exporter base class to output model to onnx, torch_script, graphdef, etc.
    """

    def __init__(self):
        self.model = None

    @classmethod
    def from_model(cls, model: Model, **kwargs):
        """Build the Exporter instance.

        @param model: A model instance. it will be used to output the generated file,
            and the configuration.json in its model_dir field will be used to create the exporter instance.
        @param kwargs: Extra kwargs used to create the Exporter instance.
        @return: The Exporter instance
        """
        cfg = Config.from_file(
            os.path.join(model.model_dir, ModelFile.CONFIGURATION))
        task_name = cfg.task
        model_cfg = cfg.model
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type
        export_cfg = ConfigDict({'type': model_cfg.type})
        if hasattr(cfg, 'export'):
            export_cfg.update(cfg.export)
        exporter = build_exporter(export_cfg, task_name, kwargs)
        exporter.model = model
        return exporter

    @abstractmethod
    def export_onnx(self, outputs: str, opset=11, **kwargs):
        """Export the model as onnx format files.

        In some cases,  several files may be generated,
        So please return a dict which contains the generated name with the file path.

        @param opset: The version of the ONNX operator set to use.
        @param outputs: The output dir.
        @param kwargs: In this default implementation,
            kwargs will be carried to generate_dummy_inputs as extra arguments (like input shape).
        @return: A dict contains the model name with the model file path.
        """
        pass

# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Union

from maas_lib.models.base import Model
from maas_lib.utils.config import ConfigDict
from maas_lib.utils.constant import Tasks
from maas_lib.utils.registry import Registry, build_from_cfg
from .base import Pipeline

PIPELINES = Registry('pipelines')


def build_pipeline(cfg: ConfigDict,
                   task_name: str = None,
                   default_args: dict = None):
    """ build pipeline given model config dict.

    Args:
        cfg (:obj:`ConfigDict`): config dict for model object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, PIPELINES, group_key=task_name, default_args=default_args)


def pipeline(task: str = None,
             model: Union[str, Model] = None,
             config_file: str = None,
             pipeline_name: str = None,
             framework: str = None,
             device: int = -1,
             **kwargs) -> Pipeline:
    """ Factory method to build a obj:`Pipeline`.


    Args:
        task (str): Task name defining which pipeline will be returned.
        model (str or obj:`Model`): model name or model object.
        config_file (str, optional): path to config file.
        pipeline_name (str, optional): pipeline class name or alias name.
        framework (str, optional): framework type.
        device (int, optional): which device is used to do inference.

    Return:
        pipeline (obj:`Pipeline`): pipeline object for certain task.

    Examples:
    ```python
    >>> p = pipeline('image-classification')
    >>> p = pipeline('text-classification', model='distilbert-base-uncased')
    >>>  # Using model object
    >>> resnet = Model.from_pretrained('Resnet')
    >>> p = pipeline('image-classification', model=resnet)
    """
    if task is not None and model is None and pipeline_name is None:
        # get default pipeline for this task
        assert task in PIPELINES.modules, f'No pipeline is registerd for Task {task}'
        pipeline_name = list(PIPELINES.modules[task].keys())[0]

    if pipeline_name is not None:
        cfg = dict(type=pipeline_name, **kwargs)
        return build_pipeline(cfg, task_name=task)

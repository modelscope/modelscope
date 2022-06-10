# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Union

import json
from maas_hub.file_download import model_file_download

from modelscope.models.base import Model
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import CONFIGFILE, Tasks
from modelscope.utils.registry import Registry, build_from_cfg
from .base import Pipeline
from .util import is_model_name

PIPELINES = Registry('pipelines')

DEFAULT_MODEL_FOR_PIPELINE = {
    # TaskName: (pipeline_module_name, model_repo)
    Tasks.image_matting: ('image-matting', 'damo/image-matting-person'),
    Tasks.text_classification:
    ('bert-sentiment-analysis', 'damo/bert-base-sst2'),
    Tasks.text_generation: ('palm', 'damo/nlp_palm_text-generation_chinese'),
    Tasks.image_captioning: ('ofa', None),
    Tasks.image_generation:
    ('person-image-cartoon',
     'damo/cv_unet_person-image-cartoon_compound-models'),
}


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
             preprocessor=None,
             config_file: str = None,
             pipeline_name: str = None,
             framework: str = None,
             device: int = -1,
             **kwargs) -> Pipeline:
    """ Factory method to build a obj:`Pipeline`.


    Args:
        task (str): Task name defining which pipeline will be returned.
        model (str or obj:`Model`): model name or model object.
        preprocessor: preprocessor object.
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
    if task is None and pipeline_name is None:
        raise ValueError('task or pipeline_name is required')

    if pipeline_name is None:
        # get default pipeline for this task
        pipeline_name, default_model_repo = get_default_pipeline_info(task)
        if model is None:
            model = default_model_repo

    assert isinstance(model, (type(None), str, Model)), \
        f'model should be either None, str or Model, but got {type(model)}'

    cfg = ConfigDict(type=pipeline_name, model=model)

    if kwargs:
        cfg.update(kwargs)

    if preprocessor is not None:
        cfg.preprocessor = preprocessor

    return build_pipeline(cfg, task_name=task)


def add_default_pipeline_info(task: str,
                              model_name: str,
                              modelhub_name: str = None,
                              overwrite: bool = False):
    """ Add default model for a task.

    Args:
        task (str): task name.
        model_name (str): model_name.
        modelhub_name (str): name for default modelhub.
        overwrite (bool): overwrite default info.
    """
    if not overwrite:
        assert task not in DEFAULT_MODEL_FOR_PIPELINE, \
            f'task {task} already has default model.'

    DEFAULT_MODEL_FOR_PIPELINE[task] = (model_name, modelhub_name)


def get_default_pipeline_info(task):
    """ Get default info for certain task.

    Args:
        task (str): task name.

    Return:
        A tuple: first element is pipeline name(model_name), second element
            is modelhub name.
    """

    if task not in DEFAULT_MODEL_FOR_PIPELINE:
        # support pipeline which does not register default model
        pipeline_name = list(PIPELINES.modules[task].keys())[0]
        default_model = None
    else:
        pipeline_name, default_model = DEFAULT_MODEL_FOR_PIPELINE[task]
    return pipeline_name, default_model

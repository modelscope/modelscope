# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import List, Union

from modelscope.models.base import Model
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import Tasks
from modelscope.utils.registry import Registry, build_from_cfg
from .base import Pipeline

PIPELINES = Registry('pipelines')

DEFAULT_MODEL_FOR_PIPELINE = {
    # TaskName: (pipeline_module_name, model_repo)
    Tasks.sentence_similarity:
    ('sbert-base-chinese-sentence-similarity',
     'damo/nlp_structbert_sentence-similarity_chinese-base'),
    Tasks.image_matting: ('image-matting', 'damo/cv_unet_image-matting_damo'),
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
             model: Union[str, List[str], Model, List[Model]] = None,
             preprocessor=None,
             config_file: str = None,
             pipeline_name: str = None,
             framework: str = None,
             device: int = -1,
             **kwargs) -> Pipeline:
    """ Factory method to build a obj:`Pipeline`.


    Args:
        task (str): Task name defining which pipeline will be returned.
        model (str or List[str] or obj:`Model` or obj:list[`Model`]): (list of) model name or model object.
        preprocessor: preprocessor object.
        config_file (str, optional): path to config file.
        pipeline_name (str, optional): pipeline class name or alias name.
        framework (str, optional): framework type.
        device (int, optional): which device is used to do inference.

    Return:
        pipeline (obj:`Pipeline`): pipeline object for certain task.

    Examples:
    ```python
    >>> # Using default model for a task
    >>> p = pipeline('image-classification')
    >>> # Using pipeline with a model name
    >>> p = pipeline('text-classification', model='damo/distilbert-base-uncased')
    >>> # Using pipeline with a model object
    >>> resnet = Model.from_pretrained('Resnet')
    >>> p = pipeline('image-classification', model=resnet)
    >>> # Using pipeline with a list of model names
    >>> p = pipeline('audio-kws', model=['damo/audio-tts', 'damo/auto-tts2'])
    """
    if task is None and pipeline_name is None:
        raise ValueError('task or pipeline_name is required')

    if pipeline_name is None:
        # get default pipeline for this task
        if isinstance(model, str) \
           or (isinstance(model, list) and isinstance(model[0], str)):

            # if is_model_name(model):
            if (isinstance(model, str) and model.startswith('damo/')) \
               or (isinstance(model, list) and model[0].startswith('damo/')) \
               or (isinstance(model, str) and osp.exists(model)):
                # TODO @wenmeng.zwm  add support when model is a str of modelhub address
                # read pipeline info from modelhub configuration file.
                pipeline_name, default_model_repo = get_default_pipeline_info(
                    task)
            else:
                pipeline_name = get_pipeline_by_model_name(task, model)
        else:
            pipeline_name, default_model_repo = get_default_pipeline_info(task)

        if model is None:
            model = default_model_repo

    assert isinstance(model, (type(None), str, Model, list)), \
        f'model should be either None, str, List[str], Model, or List[Model], but got {type(model)}'

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


def get_pipeline_by_model_name(task: str, model: Union[str, List[str]]):
    """ Get pipeline name by task name and model name

    Args:
        task (str): task name.
        model (str| list[str]): model names
    """
    if isinstance(model, str):
        model_key = model
    else:
        model_key = '_'.join(model)
    assert model_key in PIPELINES.modules[task], \
        f'pipeline for task {task} model {model_key} not found.'
    return model_key

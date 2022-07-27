# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import List, Optional, Union

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.utils.config import ConfigDict, check_config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Tasks
from modelscope.utils.hub import read_config
from modelscope.utils.registry import Registry, build_from_cfg
from .base import Pipeline
from .util import is_official_hub_path

PIPELINES = Registry('pipelines')

DEFAULT_MODEL_FOR_PIPELINE = {
    # TaskName: (pipeline_module_name, model_repo)
    Tasks.word_segmentation:
    (Pipelines.word_segmentation,
     'damo/nlp_structbert_word-segmentation_chinese-base'),
    Tasks.named_entity_recognition:
    (Pipelines.named_entity_recognition,
     'damo/nlp_transformercrf_named-entity-recognition_chinese-base-news'),
    Tasks.sentence_similarity:
    (Pipelines.sentence_similarity,
     'damo/nlp_structbert_sentence-similarity_chinese-base'),
    Tasks.translation: (Pipelines.csanmt_translation,
                        'damo/nlp_csanmt_translation'),
    Tasks.nli: (Pipelines.nli, 'damo/nlp_structbert_nli_chinese-base'),
    Tasks.sentiment_classification:
    (Pipelines.sentiment_classification,
     'damo/nlp_structbert_sentiment-classification_chinese-base'
     ),  # TODO: revise back after passing the pr
    Tasks.image_matting: (Pipelines.image_matting,
                          'damo/cv_unet_image-matting'),
    Tasks.image_denoise: (Pipelines.image_denoise,
                          'damo/cv_nafnet_image-denoise_sidd'),
    Tasks.text_classification: (Pipelines.sentiment_analysis,
                                'damo/bert-base-sst2'),
    Tasks.text_generation: (Pipelines.text_generation,
                            'damo/nlp_palm2.0_text-generation_chinese-base'),
    Tasks.zero_shot_classification:
    (Pipelines.zero_shot_classification,
     'damo/nlp_structbert_zero-shot-classification_chinese-base'),
    Tasks.dialog_intent_prediction:
    (Pipelines.dialog_intent_prediction,
     'damo/nlp_space_dialog-intent-prediction'),
    Tasks.dialog_modeling: (Pipelines.dialog_modeling,
                            'damo/nlp_space_dialog-modeling'),
    Tasks.dialog_state_tracking: (Pipelines.dialog_state_tracking,
                                  'damo/nlp_space_dialog-state-tracking'),
    Tasks.image_captioning: (Pipelines.image_captioning,
                             'damo/ofa_image-caption_coco_large_en'),
    Tasks.image_generation:
    (Pipelines.person_image_cartoon,
     'damo/cv_unet_person-image-cartoon_compound-models'),
    Tasks.ocr_detection: (Pipelines.ocr_detection,
                          'damo/cv_resnet18_ocr-detection-line-level_damo'),
    Tasks.fill_mask: (Pipelines.fill_mask, 'damo/nlp_veco_fill-mask-large'),
    Tasks.action_recognition: (Pipelines.action_recognition,
                               'damo/cv_TAdaConv_action-recognition'),
    Tasks.multi_modal_embedding:
    (Pipelines.multi_modal_embedding,
     'damo/multi-modal_clip-vit-large-patch14-chinese_multi-modal-embedding'),
    Tasks.generative_multi_modal_embedding:
    (Pipelines.generative_multi_modal_embedding,
     'damo/multi-modal_gemm-vit-large-patch14_generative-multi-modal-embedding'
     ),
    Tasks.visual_question_answering:
    (Pipelines.visual_question_answering,
     'damo/mplug_visual-question-answering_coco_large_en'),
    Tasks.video_embedding: (Pipelines.cmdssl_video_embedding,
                            'damo/cv_r2p1d_video_embedding'),
    Tasks.text_to_image_synthesis:
    (Pipelines.text_to_image_synthesis,
     'damo/cv_imagen_text-to-image-synthesis_tiny'),
    Tasks.video_multi_modal_embedding:
    (Pipelines.video_multi_modal_embedding,
     'damo/multi_modal_clip_vtretrival_msrvtt_53'),
    Tasks.image_color_enhance: (Pipelines.image_color_enhance,
                                'damo/cv_csrnet_image-color-enhance-models'),
    Tasks.virtual_tryon: (Pipelines.virtual_tryon,
                          'damo/cv_daflow_virtual-tryon_base'),
    Tasks.image_colorization: (Pipelines.image_colorization,
                               'damo/cv_unet_image-colorization'),
    Tasks.image_segmentation:
    (Pipelines.image_instance_segmentation,
     'damo/cv_swin-b_image-instance-segmentation_coco'),
    Tasks.style_transfer: (Pipelines.style_transfer,
                           'damo/cv_aams_style-transfer_damo'),
    Tasks.face_image_generation: (Pipelines.face_image_generation,
                                  'damo/cv_gan_face-image-generation'),
    Tasks.image_super_resolution: (Pipelines.image_super_resolution,
                                   'damo/cv_rrdb_image-super-resolution'),
}


def normalize_model_input(model, model_revision):
    """ normalize the input model, to ensure that a model str is a valid local path: in other words,
    for model represented by a model id, the model shall be downloaded locally
    """
    if isinstance(model, str) and is_official_hub_path(model, model_revision):
        # skip revision download if model is a local directory
        if not os.path.exists(model):
            # note that if there is already a local copy, snapshot_download will check and skip downloading
            model = snapshot_download(model, revision=model_revision)
    elif isinstance(model, list) and isinstance(model[0], str):
        for idx in range(len(model)):
            if is_official_hub_path(
                    model[idx],
                    model_revision) and not os.path.exists(model[idx]):
                model[idx] = snapshot_download(
                    model[idx], revision=model_revision)
    return model


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
             device: str = 'gpu',
             model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
             **kwargs) -> Pipeline:
    """ Factory method to build an obj:`Pipeline`.


    Args:
        task (str): Task name defining which pipeline will be returned.
        model (str or List[str] or obj:`Model` or obj:list[`Model`]): (list of) model name or model object.
        preprocessor: preprocessor object.
        config_file (str, optional): path to config file.
        pipeline_name (str, optional): pipeline class name or alias name.
        framework (str, optional): framework type.
        model_revision: revision of model(s) if getting from model hub, for multiple models, expecting
        all models to have the same revision
        device (str, optional): whether to use gpu or cpu is used to do inference.

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

    assert isinstance(model, (type(None), str, Model, list)), \
        f'model should be either None, str, List[str], Model, or List[Model], but got {type(model)}'

    model = normalize_model_input(model, model_revision)

    if pipeline_name is None:
        # get default pipeline for this task
        if isinstance(model, str) \
           or (isinstance(model, list) and isinstance(model[0], str)):
            if is_official_hub_path(model, revision=model_revision):
                # read config file from hub and parse
                cfg = read_config(
                    model, revision=model_revision) if isinstance(
                        model, str) else read_config(
                            model[0], revision=model_revision)
                check_config(cfg)
                pipeline_name = cfg.pipeline.type
            else:
                # used for test case, when model is str and is not hub path
                pipeline_name = get_pipeline_by_model_name(task, model)
        elif isinstance(model, Model) or \
                (isinstance(model, list) and isinstance(model[0], Model)):
            # get pipeline info from Model object
            first_model = model[0] if isinstance(model, list) else model
            if not hasattr(first_model, 'pipeline'):
                # model is instantiated by user, we should parse config again
                cfg = read_config(first_model.model_dir)
                check_config(cfg)
                first_model.pipeline = cfg.pipeline
            pipeline_name = first_model.pipeline.type
        else:
            pipeline_name, default_model_repo = get_default_pipeline_info(task)
            model = normalize_model_input(default_model_repo, model_revision)

    cfg = ConfigDict(type=pipeline_name, model=model)
    cfg.device = device
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

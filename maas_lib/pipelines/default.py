# Copyright (c) Alibaba, Inc. and its affiliates.

from maas_lib.utils.constant import Tasks

DEFAULT_MODEL_FOR_PIPELINE = {
    # TaskName: (pipeline_module_name, model_repo)
    Tasks.image_matting: ('image-matting', 'damo/image-matting-person'),
    Tasks.text_classification:
    ('bert-sentiment-analysis', 'damo/bert-base-sst2'),
    Tasks.text_generation: ('palm', 'damo/nlp_palm_text-generation_chinese'),
    Tasks.image_captioning: ('ofa', None),
}


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
    assert task in DEFAULT_MODEL_FOR_PIPELINE, \
        f'No default pipeline is registered for Task {task}'

    pipeline_name, default_model = DEFAULT_MODEL_FOR_PIPELINE[task]
    return pipeline_name, default_model

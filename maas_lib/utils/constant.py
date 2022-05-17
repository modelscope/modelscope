# Copyright (c) Alibaba, Inc. and its affiliates.


class Fields(object):
    """ Names for different application fields
    """
    image = 'image'
    video = 'video'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi_modal'


class Tasks(object):
    """ Names for tasks supported by maas lib.

    Holds the standard task name to use for identifying different tasks.
    This should be used to register models, pipelines, trainers.
    """
    # vision tasks
    image_classfication = 'image-classification'
    object_detection = 'object-detection'

    # nlp tasks
    sentiment_analysis = 'sentiment-analysis'
    fill_mask = 'fill-mask'


class InputFields(object):
    """ Names for input data fileds in the input data for pipelines
    """
    img = 'img'
    text = 'text'
    audio = 'audio'

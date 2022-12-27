# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Define TTS exceptions
"""


class TtsException(Exception):
    """
    TTS exception class.
    """
    pass


class TtsModelConfigurationException(TtsException):
    """
    TTS model configuration exceptions.
    """
    pass


class TtsModelNotExistsException(TtsException):
    """
    TTS model not exists exception.
    """


class TtsVoiceNotExistsException(TtsException):
    """
    TTS voice not exists exception.
    """
    pass


class TtsFrontendException(TtsException):
    """
    TTS frontend module level exceptions.
    """
    pass


class TtsFrontendInitializeFailedException(TtsFrontendException):
    """
    If tts frontend resource is invalid or not exist, this exception will be raised.
    """
    pass


class TtsFrontendLanguageTypeInvalidException(TtsFrontendException):
    """
    If language type is invalid, this exception will be raised.
    """


class TtsVocoderException(TtsException):
    """
    Vocoder exception
    """


class TtsVocoderMelspecShapeMismatchException(TtsVocoderException):
    """
    If vocoder's input melspec shape mismatch, this exception will be raised.
    """


class TtsDataPreprocessorException(TtsException):
    """
    Tts data preprocess exception
    """


class TtsDataPreprocessorDirNotExistsException(TtsDataPreprocessorException):
    """
    If any dir is not exists, this exception will be raised.
    """


class TtsDataPreprocessorAudioConfigNotExistsException(
        TtsDataPreprocessorException):
    """
    If audio config is not exists, this exception will be raised.
    """


class TtsTrainingException(TtsException):
    """
    Tts training exception
    """


class TtsTrainingHparamsInvalidException(TtsException):
    """
    If training hparams is invalid, this exception will be raised.
    """


class TtsTrainingWorkDirNotExistsException(TtsTrainingException):
    """
    If training work dir not exists, this exception will be raised.
    """


class TtsTrainingCfgNotExistsException(TtsTrainingException):
    """
    If training cfg not exists, this exception will be raised.
    """


class TtsTrainingDatasetInvalidException(TtsTrainingException):
    """
    If dataset invalid, this exception will be raised.
    """


class TtsTrainingInvalidModelException(TtsTrainingException):
    """
    If model is invalid or not exists, this exception will be raised.
    """

"""
Define TTS exceptions
"""


class TtsException(Exception):
    """
    TTS exception class.
    """
    pass


class TtsModelConfigurationExcetion(TtsException):
    """
    TTS model configuration exceptions.
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

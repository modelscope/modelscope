# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import sys
from functools import partial
from pathlib import Path
import importlib
from types import MethodType
from typing import BinaryIO, Dict, List, Optional, Union

from huggingface_hub.hf_api import CommitInfo, future_compatible
from modelscope import snapshot_download
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke
from modelscope.utils.logger import get_logger

try:
    from transformers import AutoModelForImageToImage as AutoModelForImageToImageHF

    AutoModelForImageToImage = get_wrapped_class(AutoModelForImageToImageHF)
except ImportError:
    AutoModelForImageToImage = UnsupportedAutoClass('AutoModelForImageToImage')

try:
    from transformers import AutoModelForImageTextToText as AutoModelForImageTextToTextHF

    AutoModelForImageTextToText = get_wrapped_class(
        AutoModelForImageTextToTextHF)
except ImportError:
    AutoModelForImageTextToText = UnsupportedAutoClass(
        'AutoModelForImageTextToText')

try:
    from transformers import AutoModelForKeypointDetection as AutoModelForKeypointDetectionHF

    AutoModelForKeypointDetection = get_wrapped_class(
        AutoModelForKeypointDetectionHF)
except ImportError:
    AutoModelForKeypointDetection = UnsupportedAutoClass(
        'AutoModelForKeypointDetection')

try:
    from transformers import \
        Qwen2VLForConditionalGeneration as Qwen2VLForConditionalGenerationHF

    Qwen2VLForConditionalGeneration = get_wrapped_class(
        Qwen2VLForConditionalGenerationHF)
except ImportError:
    Qwen2VLForConditionalGeneration = UnsupportedAutoClass(
        'Qwen2VLForConditionalGeneration')


logger = get_logger()


def get_wrapped_class(module_class,
                      ignore_file_pattern=[],
                      file_filter=None,
                      **kwargs):
    """Get a custom wrapper class for  auto classes to download the models from the ModelScope hub
    Args:
        module_class: The actual module class
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
    Returns:
        The wrapper
    """
    default_ignore_file_pattern = ignore_file_pattern
    default_file_filter = file_filter

    class ClassWrapper(module_class):

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            ignore_file_pattern = kwargs.pop('ignore_file_pattern',
                                             default_ignore_file_pattern)
            subfolder = kwargs.pop('subfolder', default_file_filter)
            file_filter = None
            if subfolder:
                file_filter = f'{subfolder}/*'
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', DEFAULT_MODEL_REVISION)
                if file_filter is None:
                    model_dir = snapshot_download(
                        pretrained_model_name_or_path,
                        revision=revision,
                        ignore_file_pattern=ignore_file_pattern,
                        user_agent=user_agent())
                else:
                    model_dir = os.path.join(
                        snapshot_download(
                            pretrained_model_name_or_path,
                            revision=revision,
                            ignore_file_pattern=ignore_file_pattern,
                            allow_file_pattern=file_filter,
                            user_agent=user_agent()), subfolder)
            else:
                model_dir = pretrained_model_name_or_path

            module_obj = module_class.from_pretrained(model_dir, *model_args,
                                                      **kwargs)

            if module_class.__name__.startswith('AutoModel'):
                module_obj.model_dir = model_dir
            return module_obj

    ClassWrapper.__name__ = module_class.__name__
    ClassWrapper.__qualname__ = module_class.__qualname__
    return ClassWrapper


AutoModel = get_wrapped_class(AutoModelHF)
AutoModelForCausalLM = get_wrapped_class(AutoModelForCausalLMHF)
AutoModelForSeq2SeqLM = get_wrapped_class(AutoModelForSeq2SeqLMHF)
AutoModelForVision2Seq = get_wrapped_class(AutoModelForVision2SeqHF)
AutoModelForSequenceClassification = get_wrapped_class(
    AutoModelForSequenceClassificationHF)
AutoModelForTokenClassification = get_wrapped_class(
    AutoModelForTokenClassificationHF)
AutoModelForImageSegmentation = get_wrapped_class(
    AutoModelForImageSegmentationHF)
AutoModelForImageClassification = get_wrapped_class(
    AutoModelForImageClassificationHF)
AutoModelForZeroShotImageClassification = get_wrapped_class(
    AutoModelForZeroShotImageClassificationHF)
AutoModelForQuestionAnswering = get_wrapped_class(
    AutoModelForQuestionAnsweringHF)
AutoModelForTableQuestionAnswering = get_wrapped_class(
    AutoModelForTableQuestionAnsweringHF)
AutoModelForVisualQuestionAnswering = get_wrapped_class(
    AutoModelForVisualQuestionAnsweringHF)
AutoModelForDocumentQuestionAnswering = get_wrapped_class(
    AutoModelForDocumentQuestionAnsweringHF)
AutoModelForSemanticSegmentation = get_wrapped_class(
    AutoModelForSemanticSegmentationHF)
AutoModelForUniversalSegmentation = get_wrapped_class(
    AutoModelForUniversalSegmentationHF)
AutoModelForInstanceSegmentation = get_wrapped_class(
    AutoModelForInstanceSegmentationHF)
AutoModelForObjectDetection = get_wrapped_class(AutoModelForObjectDetectionHF)
AutoModelForZeroShotObjectDetection = get_wrapped_class(
    AutoModelForZeroShotObjectDetectionHF)
AutoModelForAudioClassification = get_wrapped_class(
    AutoModelForAudioClassificationHF)
AutoModelForSpeechSeq2Seq = get_wrapped_class(AutoModelForSpeechSeq2SeqHF)
AutoModelForMaskedImageModeling = get_wrapped_class(
    AutoModelForMaskedImageModelingHF)
AutoModelForMaskedLM = get_wrapped_class(AutoModelForMaskedLMHF)
AutoModelForMaskGeneration = get_wrapped_class(AutoModelForMaskGenerationHF)
AutoModelForPreTraining = get_wrapped_class(AutoModelForPreTrainingHF)
AutoModelForTextEncoding = get_wrapped_class(AutoModelForTextEncodingHF)
T5EncoderModel = get_wrapped_class(T5EncoderModelHF)

AutoTokenizer = get_wrapped_class(
    AutoTokenizerHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
AutoProcessor = get_wrapped_class(
    AutoProcessorHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
AutoConfig = get_wrapped_class(
    AutoConfigHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
GenerationConfig = get_wrapped_class(
    GenerationConfigHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
BitsAndBytesConfig = get_wrapped_class(
    BitsAndBytesConfigHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])
AutoImageProcessor = get_wrapped_class(
    AutoImageProcessorHF,
    ignore_file_pattern=[
        r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth', r'\w+\.pt', r'\w+\.h5'
    ])

GPTQConfig = GPTQConfigHF
AwqConfig = AwqConfigHF
BatchFeature = get_wrapped_class(BatchFeatureHF)

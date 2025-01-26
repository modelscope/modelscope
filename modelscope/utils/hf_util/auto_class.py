# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import __version__ as transformers_version

    from transformers import AutoConfig
    from transformers import AutoFeatureExtractor
    from transformers import AutoImageProcessor
    from transformers import AutoModel
    from transformers import AutoModelForAudioClassification
    from transformers import AutoModelForCausalLM
    from transformers import AutoModelForDocumentQuestionAnswering
    from transformers import AutoModelForImageClassification
    from transformers import AutoModelForImageSegmentation
    from transformers import AutoModelForInstanceSegmentation
    from transformers import AutoModelForMaskedImageModeling
    from transformers import AutoModelForMaskedLM
    from transformers import AutoModelForMaskGeneration
    from transformers import AutoModelForObjectDetection
    from transformers import AutoModelForPreTraining
    from transformers import AutoModelForQuestionAnswering
    from transformers import AutoModelForSemanticSegmentation
    from transformers import AutoModelForSeq2SeqLM
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoModelForSpeechSeq2Seq
    from transformers import AutoModelForTableQuestionAnswering
    from transformers import AutoModelForTextEncoding
    from transformers import AutoModelForTokenClassification
    from transformers import AutoModelForUniversalSegmentation
    from transformers import AutoModelForVision2Seq
    from transformers import AutoModelForVisualQuestionAnswering
    from transformers import AutoModelForZeroShotImageClassification
    from transformers import AutoModelForZeroShotObjectDetection
    from transformers import AutoProcessor
    from transformers import AutoTokenizer
    from transformers import BatchFeature
    from transformers import BitsAndBytesConfig
    from transformers import GenerationConfig
    from transformers import (PretrainedConfig, PreTrainedModel,
                              PreTrainedTokenizerBase)
    from transformers import T5EncoderModel

    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        pass

    try:
        from transformers import GPTQConfig
        from transformers import AwqConfig
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageToImage
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        pass

    try:
        from transformers import AutoModelForKeypointDetection
    except ImportError:
        pass

else:

    class UnsupportedAutoClass:

        def __init__(self, name: str):
            self.error_msg =\
                f'{name} is not supported with your installed Transformers version {transformers_version}. ' + \
                'Please update your Transformers by "pip install transformers -U".'

        def from_pretrained(self, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            raise ImportError(self.error_msg)

        def from_config(self, cls, config):
            raise ImportError(self.error_msg)

    def user_agent(invoked_by=None):
        from modelscope.utils.constant import Invoke

        if invoked_by is None:
            invoked_by = Invoke.PRETRAINED
        uagent = '%s/%s' % (Invoke.KEY, invoked_by)
        return uagent

    from .patcher import get_all_imported_modules, _patch_pretrained_class

    all_imported_modules = get_all_imported_modules()
    all_available_modules = _patch_pretrained_class(all_imported_modules, wrap=True)

    for module in all_available_modules:
        globals()[module.__name__] = module

# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from transformers import LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM

    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None

    try:
        from transformers import GPTQConfig
        from transformers import AwqConfig
    except ImportError:
        GPTQConfig = None
        AwqConfig = None

    try:
        from transformers import AutoModelForImageToImage
    except ImportError:
        AutoModelForImageToImage = None

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoModelForKeypointDetection
    except ImportError:
        AutoModelForKeypointDetection = None

else:

    from .patcher import get_all_imported_modules, _patch_pretrained_class
    all_available_modules = _patch_pretrained_class(
        get_all_imported_modules(), wrap=True)

    for module in all_available_modules:
        globals()[module.__name__] = module

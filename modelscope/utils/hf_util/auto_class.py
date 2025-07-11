# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import (AutoConfig, AutoFeatureExtractor,
                              AutoImageProcessor, AutoModel,
                              AutoModelForAudioClassification,
                              AutoModelForCausalLM,
                              AutoModelForDocumentQuestionAnswering,
                              AutoModelForImageClassification,
                              AutoModelForImageSegmentation,
                              AutoModelForInstanceSegmentation,
                              AutoModelForMaskedImageModeling,
                              AutoModelForMaskedLM, AutoModelForMaskGeneration,
                              AutoModelForObjectDetection,
                              AutoModelForPreTraining,
                              AutoModelForQuestionAnswering,
                              AutoModelForSemanticSegmentation,
                              AutoModelForSeq2SeqLM,
                              AutoModelForSequenceClassification,
                              AutoModelForSpeechSeq2Seq,
                              AutoModelForTableQuestionAnswering,
                              AutoModelForTextEncoding,
                              AutoModelForTokenClassification,
                              AutoModelForUniversalSegmentation,
                              AutoModelForVision2Seq,
                              AutoModelForVisualQuestionAnswering,
                              AutoModelForZeroShotImageClassification,
                              AutoModelForZeroShotObjectDetection,
                              AutoProcessor, AutoTokenizer, BatchFeature,
                              BitsAndBytesConfig, GenerationConfig,
                              LlamaForCausalLM, LlamaModel,
                              LlamaPreTrainedModel, PretrainedConfig,
                              PreTrainedModel, PreTrainedTokenizerBase,
                              T5EncoderModel)

    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None

    try:
        from transformers import AwqConfig, GPTQConfig
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
    pass

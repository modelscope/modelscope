# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .ans_pipeline import ANSPipeline
    from .asr_inference_pipeline import AutomaticSpeechRecognitionPipeline
    from .kws_farfield_pipeline import KWSFarfieldPipeline
    from .kws_kwsbp_pipeline import KeyWordSpottingKwsbpPipeline
    from .linear_aec_pipeline import LinearAECPipeline
    from .text_to_speech_pipeline import TextToSpeechSambertHifiganPipeline
    from .inverse_text_processing_pipeline import InverseTextProcessingPipeline
    from .separation_pipeline import SeparationPipeline
    from .speaker_verification_pipeline import SpeakerVerificationPipeline
    from .ssr_pipeline import SSRPipeline
    from .voice_conversion_pipeline import VCPipeline
else:
    _import_structure = {
        'ans_dfsmn_pipeline': ['ANSDFSMNPipeline'],
        'ans_pipeline': ['ANSPipeline'],
        'asr_inference_pipeline': ['AutomaticSpeechRecognitionPipeline'],
        'kws_farfield_pipeline': ['KWSFarfieldPipeline'],
        'kws_kwsbp_pipeline': ['KeyWordSpottingKwsbpPipeline'],
        'linear_aec_pipeline': ['LinearAECPipeline'],
        'text_to_speech_pipeline': ['TextToSpeechSambertHifiganPipeline'],
        'itn_inference_pipeline': ['InverseTextProcessingPipeline'],
        'inverse_text_processing_pipeline': ['InverseTextProcessingPipeline'],
        'separation_pipeline': ['SeparationPipeline'],
        'speaker_verification_pipeline': ['SpeakerVerificationPipeline'],
        'speech-super-resolution-inference': ['SSRPipeline'],
        'voice_conversion': ['VCPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

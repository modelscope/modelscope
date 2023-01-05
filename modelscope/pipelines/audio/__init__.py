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
else:
    _import_structure = {
        'ans_pipeline': ['ANSPipeline'],
        'asr_inference_pipeline': ['AutomaticSpeechRecognitionPipeline'],
        'kws_farfield_pipeline': ['KWSFarfieldPipeline'],
        'kws_kwsbp_pipeline': ['KeyWordSpottingKwsbpPipeline'],
        'linear_aec_pipeline': ['LinearAECPipeline'],
        'text_to_speech_pipeline': ['TextToSpeechSambertHifiganPipeline'],
        'itn_inference_pipeline': ['InverseTextProcessingPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

from modelscope.utils.error import AUDIO_IMPORT_ERROR
from .base import Pipeline
from .builder import pipeline
from .cv import *  # noqa F403
from .multi_modal import *  # noqa F403
from .nlp import *  # noqa F403

try:
    from .audio import LinearAECPipeline
    from .audio.ans_pipeline import ANSPipeline
except ModuleNotFoundError as e:
    print(AUDIO_IMPORT_ERROR.format(e))

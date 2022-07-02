try:
    from .kws_kwsbp_pipeline import *  # noqa F403
    from .linear_aec_pipeline import LinearAECPipeline
except ModuleNotFoundError as e:
    if str(e) == "No module named 'torch'":
        pass
    else:
        raise ModuleNotFoundError(e)

try:
    from .text_to_speech_pipeline import *  # noqa F403
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        pass
    else:
        raise ModuleNotFoundError(e)

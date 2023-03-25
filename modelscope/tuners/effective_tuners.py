import torch
from typing import List, Union


class TunerType:

    LORA = 'lora'
    PROMPT_TUNING = 'prompt_tuning'
    DREAM_BOOTH = 'dream_booth'


def initialize_tuner(model: torch.nn.Module, tuner_type: Union[str, List[str]], **kwargs):
    if isinstance(tuner_type, str):
        tuner_type = [tuner_type]

    for _type in tuner_type:
        if _type == TunerType.LORA:
            pass
        elif _type == TunerType.DREAM_BOOTH:
            pass
        elif _type == TunerType.PROMPT_TUNING:
            pass




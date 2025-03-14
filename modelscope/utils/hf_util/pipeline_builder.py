import os
from typing import Optional, Union

from modelscope.hub import snapshot_download
from modelscope.utils.hf_util.patcher import _patch_pretrained_class


def _get_hf_device(device):
    if isinstance(device, str):
        device_name = device.lower()
        eles = device_name.split(':')
        if eles[0] == 'gpu':
            eles = ['cuda'] + eles[1:]
            device = ''.join(eles)
    return device


def _get_hf_pipeline_class(task, model):
    from transformers.pipelines import check_task, get_task
    if not task:
        task = get_task(model)
    normalized_task, targeted_task, task_options = check_task(task)
    pipeline_class = targeted_task['impl']
    pipeline_class = _patch_pretrained_class([pipeline_class])[0]
    return pipeline_class


def hf_pipeline(
    task: str = None,
    model: Optional[Union[str, 'PreTrainedModel', 'TFPreTrainedModel']] = None,
    framework: Optional[str] = None,
    device: Optional[Union[int, str, 'torch.device']] = None,
    **kwargs,
) -> 'transformers.Pipeline':
    from transformers import pipeline
    if isinstance(model, str):
        if not os.path.exists(model):
            model = snapshot_download(model)

    framework = 'pt' if framework == 'pytorch' else framework

    device = _get_hf_device(device)
    pipeline_class = _get_hf_pipeline_class(task, model)

    kwargs.pop('external_engine_for_llm', None)
    kwargs.pop('llm_framework', None)

    return pipeline(
        task=task,
        model=model,
        framework=framework,
        device=device,
        pipeline_class=pipeline_class,
        **kwargs)

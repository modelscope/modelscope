import os
from typing import Optional, Union

from modelscope.hub import snapshot_download
from modelscope.utils.hf_util.patcher import _patch_pretrained_class
from modelscope.utils.logger import get_logger

logger = get_logger()


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


def sentence_transformers_pipeline(model: str, **kwargs):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            'Could not import sentence_transformers, please upgrade to the latest version of sentence_transformers '
            "with: 'pip install -U sentence_transformers'") from None
    if isinstance(model, str):
        if not os.path.exists(model):
            model = snapshot_download(model)

    from modelscope.pipelines import Pipeline

    class SentenceTransformerPipeline(Pipeline):
        """A wrapper for sentence_transformers.SentenceTransformer to make it compatible
        with the modelscope pipeline conventions."""

        def __init__(self, model_path: str, **kwargs):
            self.model = SentenceTransformer(model_path, **kwargs)

        def __call__(self,
                     sentences: str | list[str] | None = None,
                     prompt_name: str | None = None,
                     **kwargs):
            input_data = kwargs.pop('input', None)
            if input_data is not None:
                sentences = input_data['source_sentence']
                res = self.model.encode(sentences, **kwargs)
                return {'text_embedding': res}
            return self.model.encode(
                sentences, prompt_name=prompt_name, **kwargs)

    return SentenceTransformerPipeline(model, **kwargs)

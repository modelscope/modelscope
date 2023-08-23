import os
from typing import Optional

from modelscope.metainfo import Tasks
from modelscope.utils.ast_utils import INDEX_KEY
from modelscope.utils.import_utils import LazyImportModule


def can_load_by_ms(model_dir: str, tast_name: str, model_type: str) -> bool:
    if ('MODELS', tast_name,
            model_type) in LazyImportModule.AST_INDEX[INDEX_KEY]:
        return True
    ms_wrapper_path = os.path.join(model_dir, 'ms_wrapper.py')
    if os.path.exists(ms_wrapper_path):
        return True
    return False


def _can_load_by_hf_automodel(automodel_class: type, config) -> bool:
    automodel_class_name = automodel_class.__name__
    if type(config) in automodel_class._model_mapping.keys():
        return True
    if hasattr(config, 'auto_map') and automodel_class_name in config.auto_map:
        return True
    return False


def get_hf_automodel_class(model_dir: str, task_name: str) -> Optional[type]:
    from modelscope import (AutoConfig, AutoModel, AutoModelForCausalLM,
                            AutoModelForSeq2SeqLM,
                            AutoModelForTokenClassification,
                            AutoModelForSequenceClassification)
    automodel_mapping = {
        Tasks.backbone: AutoModel,
        Tasks.chat: AutoModelForCausalLM,
        Tasks.text_generation: AutoModelForCausalLM,
        Tasks.text_classification: AutoModelForSequenceClassification,
        Tasks.token_classification: AutoModelForTokenClassification,
    }
    automodel_class = automodel_mapping.get(task_name, None)
    if automodel_class is None:
        return None
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    try:
        try:
            config = AutoConfig.from_pretrained(
                model_dir, trust_remote_code=True)
        except (FileNotFoundError, ValueError):
            return None

        if _can_load_by_hf_automodel(automodel_class, config):
            return automodel_class
        if (automodel_class is AutoModelForCausalLM
                and _can_load_by_hf_automodel(AutoModelForSeq2SeqLM, config)):
            return AutoModelForSeq2SeqLM
        return None
    except Exception:
        return None


def try_to_load_hf_model(model_dir: str, task_name: str,
                         use_hf: Optional[bool], **kwargs):
    automodel_class = get_hf_automodel_class(model_dir, task_name)

    if use_hf and automodel_class is None:
        raise ValueError(f'Model import failed. You used `use_hf={use_hf}`, '
                         'but the model is not a model of hf')

    model = None
    if automodel_class is not None:
        # use hf
        device_map = kwargs.get('device_map', None)
        torch_dtype = kwargs.get('torch_dtype', None)
        config = kwargs.get('config', None)

        model = automodel_class.from_pretrained(
            model_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            config=config,
            trust_remote_code=True)
    return model

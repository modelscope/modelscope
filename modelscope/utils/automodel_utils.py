import inspect
import os
from types import MethodType
from typing import Any, Optional

from modelscope.metainfo import Tasks
from modelscope.utils.ast_utils import INDEX_KEY
from modelscope.utils.import_utils import (LazyImportModule,
                                           is_transformers_available)


def can_load_by_ms(model_dir: str, task_name: Optional[str],
                   model_type: Optional[str]) -> bool:
    if model_type is None or task_name is None:
        return False
    if ('MODELS', task_name,
            model_type) in LazyImportModule.AST_INDEX[INDEX_KEY]:
        return True
    ms_wrapper_path = os.path.join(model_dir, 'ms_wrapper.py')
    if os.path.exists(ms_wrapper_path):
        return True
    return False


def fix_upgrade(module_obj: Any):
    from transformers import PreTrainedModel
    if hasattr(module_obj, '_set_gradient_checkpointing') \
            and 'value' in inspect.signature(module_obj._set_gradient_checkpointing).parameters.keys():
        module_obj._set_gradient_checkpointing = MethodType(
            PreTrainedModel._set_gradient_checkpointing, module_obj)


def post_init(self, *args, **kwargs):
    fix_upgrade(self)
    self.post_init_origin(*args, **kwargs)


def fix_transformers_upgrade():
    if is_transformers_available():
        # from 4.35.0, transformers changes its arguments of _set_gradient_checkpointing
        import transformers
        from transformers import PreTrainedModel
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.35.0') \
                and not hasattr(PreTrainedModel, 'post_init_origin'):
            PreTrainedModel.post_init_origin = PreTrainedModel.post_init
            PreTrainedModel.post_init = post_init


def _can_load_by_hf_automodel(automodel_class: type, config) -> bool:
    automodel_class_name = automodel_class.__name__
    if type(config) in automodel_class._model_mapping.keys():
        return True
    if hasattr(config, 'auto_map') and automodel_class_name in config.auto_map:
        return True
    return False


def get_default_automodel(config) -> Optional[type]:
    import modelscope.utils.hf_util as hf_util
    if not hasattr(config, 'auto_map'):
        return None
    auto_map = config.auto_map
    automodel_list = [k for k in auto_map.keys() if k.startswith('AutoModel')]
    if len(automodel_list) == 1:
        return getattr(hf_util, automodel_list[0])
    if len(automodel_list) > 1 and len(
            set([auto_map[k] for k in automodel_list])) == 1:
        return getattr(hf_util, automodel_list[0])
    return None


def get_hf_automodel_class(model_dir: str,
                           task_name: Optional[str]) -> Optional[type]:
    from modelscope.utils.hf_util import (AutoConfig, AutoModel,
                                          AutoModelForCausalLM,
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
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        if task_name is None:
            automodel_class = get_default_automodel(config)
        else:
            automodel_class = automodel_mapping.get(task_name, None)

        if automodel_class is None:
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
                         'but the model is not a model of hf.')

    model = None
    if automodel_class is not None:
        # use hf
        model = automodel_class.from_pretrained(model_dir, **kwargs)
    return model

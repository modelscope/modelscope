import inspect
import os
from functools import lru_cache
from types import MethodType
from typing import Any, List, Optional

from modelscope import get_logger
from modelscope.metainfo import Tasks
from modelscope.utils.ast_utils import INDEX_KEY
from modelscope.utils.import_utils import (LazyImportModule,
                                           is_torch_available,
                                           is_transformers_available)

logger = get_logger()


def can_load_by_ms(model_dir: str, task_name: Optional[str],
                   model_type: Optional[str]) -> bool:
    if model_type is None or task_name is None:
        return False
    if ('MODELS', task_name,
            model_type) in LazyImportModule.get_ast_index()[INDEX_KEY]:
        return True
    ms_wrapper_path = os.path.join(model_dir, 'ms_wrapper.py')
    if os.path.exists(ms_wrapper_path):
        return True
    return False


def fix_upgrade(module_obj: Any):
    from transformers import PreTrainedModel
    if hasattr(module_obj, '_set_gradient_checkpointing') \
            and 'value' in inspect.signature(
                module_obj._set_gradient_checkpointing).parameters.keys() \
            and 'modelscope.' in str(module_obj.__class__):
        module_obj._set_gradient_checkpointing = MethodType(
            PreTrainedModel._set_gradient_checkpointing, module_obj)


def post_init(self, *args, **kwargs):
    fix_upgrade(self)
    self.post_init_origin(*args, **kwargs)


def fix_transformers_upgrade():
    if is_transformers_available() and is_torch_available():
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
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
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


TRUSTED_MODEL_OWNERS = frozenset({'damo', 'iic'})


def _is_valid_hub_model_id(model_id: str) -> bool:
    if not isinstance(model_id, str) or os.path.isabs(model_id) \
            or '\\' in model_id or model_id.count('/') != 1:
        return False
    owner, name = model_id.split('/')
    return bool(owner and name and owner not in {'.', '..'}
                and name not in {'.', '..'})


@lru_cache(maxsize=256)
def _is_model_from_trusted_source(model_id: str, revision: Optional[str],
                                  trusted_owners: frozenset[str]) -> bool:
    """Verify a trusted publisher using Hub repository metadata.

    Local directory names and cache layouts are attacker-controlled and must
    not authorize remote code execution. Local paths and offline lookups remain
    untrusted because their Hub metadata cannot be verified.
    """
    if not _is_valid_hub_model_id(model_id):
        return False
    try:
        from modelscope.hub.api import HubApi
        metadata = HubApi().get_model(model_id, revision=revision)
    except Exception:
        return False

    owner = metadata.get('Owner')
    name = metadata.get('Name')
    requested_name = model_id.split('/', 1)[1]
    return (isinstance(owner, str) and owner.casefold() in trusted_owners
            and name == requested_name)


def is_model_from_trusted_source(
        model_id: str,
        revision: Optional[str] = None,
        trusted_owners: Optional[List[str]] = None) -> bool:
    """Return whether Hub metadata verifies a trusted model publisher."""
    owners = frozenset(owner.casefold()
                       for owner in (trusted_owners or TRUSTED_MODEL_OWNERS))
    return _is_model_from_trusted_source(model_id, revision, owners)


def check_model_from_owner_group(model_dir: str,
                                 owner_group: List[str] = None) -> bool:
    """Retained local-path heuristic for import compatibility only.

    Do not use this helper to authorize remote code execution. New call paths
    must use :func:`is_model_from_trusted_source` to validate Hub metadata.
    """
    if not model_dir or not isinstance(model_dir, str):
        return False
    if owner_group is None:
        owner_group = ['iic', 'damo']
    model_dir = os.path.normpath(model_dir.rstrip('/').rstrip('\\'))
    parent_dir = os.path.dirname(model_dir)
    group = os.path.basename(parent_dir)
    if group in owner_group:
        return True
    # Also check cache path pattern: {cache_root}/{owner}--{model_name}/snapshots/{revision}
    # Require exactly "{owner}--{name}" with both segments non-empty
    # to prevent spoofing via accounts like "iic--hacked" (paths like
    # "iic--hacked--evil") or empty names like "iic--".
    grandparent = os.path.basename(os.path.dirname(parent_dir))
    if '--' in grandparent:
        parts = grandparent.split('--')
        # Both owner and name must be non-empty; reject "iic--" / "--name".
        if len(parts) == 2 and all(parts) and parts[0] in owner_group:
            return True
    return False

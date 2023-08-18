# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig

from modelscope.hub.check_model import check_local_model_is_latest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Tasks
from modelscope.models.builder import build_backbone, build_model
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke, ModelFile
from modelscope.utils.device import verify_device
from modelscope.utils.logger import get_logger
from modelscope.utils.plugins import (register_modelhub_repo,
                                      register_plugins_repo)

logger = get_logger()

Tensor = Union['torch.Tensor', 'tf.Tensor']


def _can_load_by_automodel(automodel_class: type,
                           config: PretrainedConfig) -> bool:
    automodel_class_name = automodel_class.__name__
    if type(config) in automodel_class._model_mapping.keys():
        return True
    if hasattr(config, 'auto_map') and automodel_class_name in config.auto_map:
        return True
    return False


def get_automodel_class(model_dir: str, task_name: str) -> Optional[type]:
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
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except (FileNotFoundError, ValueError):
        return None

    if _can_load_by_automodel(automodel_class, config):
        return automodel_class
    if (automodel_class is AutoModelForCausalLM
            and _can_load_by_automodel(AutoModelForSeq2SeqLM, config)):
        return AutoModelForSeq2SeqLM
    return None


class Model(ABC):
    """Base model interface.
    """

    def __init__(self, model_dir, *args, **kwargs):
        self.model_dir = model_dir
        device_name = kwargs.get('device', 'gpu')
        verify_device(device_name)
        self._device_name = device_name

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.postprocess(self.forward(*args, **kwargs))

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run the forward pass for a model.

        Returns:
            Dict[str, Any]: output from the model forward pass
        """
        pass

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """ Model specific postprocess and convert model output to
        standard model outputs.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return inputs

    @classmethod
    def _instantiate(cls, **kwargs):
        """ Define the instantiation method of a model,default method is by
            calling the constructor. Note that in the case of no loading model
            process in constructor of a task model, a load_model method is
            added, and thus this method is overloaded
        """
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        cfg_dict: Config = None,
                        device: str = None,
                        **kwargs):
        """Instantiate a model from local directory or remote model repo. Note
        that when loading from remote, the model revision can be specified.

        Args:
            model_name_or_path(str): A model dir or a model id to be loaded
            revision(str, `optional`): The revision used when the model_name_or_path is
                a model id of the remote hub. default `master`.
            cfg_dict(Config, `optional`): An optional model config. If provided, it will replace
                the config read out of the `model_name_or_path`
            device(str, `optional`): The device to load the model.
            **kwargs:
                task(str, `optional`): The `Tasks` enumeration value to replace the task value
                    read out of config in the `model_name_or_path`. This is useful when the model to be loaded is not
                    equal to the model saved.
                    For example, load a `backbone` into a `text-classification` model.
                    Other kwargs will be directly fed into the `model` key, to replace the default configs.
                ignore_file_pattern(List[str], `optional`):
                    This parameter is passed to snapshot_download
                device_map(str | Dict[str, str], `optional`):
                    This parameter is passed to AutoModel or AutoModelForxxx
                torch_dtype(torch.dtype, `optional`):
                    This parameter is passed to AutoModel or AutoModelForxxx
                config(PretrainedConfig, `optional`):
                    This parameter is passed to AutoModel or AutoModelForxxx
        Returns:
            A model instance.

        Examples:
            >>> from modelscope.models import Model
            >>> Model.from_pretrained('damo/nlp_structbert_backbone_base_std', task='text-classification')
        """
        prefetched = kwargs.get('model_prefetched')
        if prefetched is not None:
            kwargs.pop('model_prefetched')
        invoked_by = kwargs.get(Invoke.KEY)
        if invoked_by is not None:
            kwargs.pop(Invoke.KEY)
        else:
            invoked_by = Invoke.PRETRAINED

        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        else:
            if prefetched is True:
                raise RuntimeError(
                    'Expecting model is pre-fetched locally, but is not found.'
                )

            invoked_by = '%s/%s' % (Invoke.KEY, invoked_by)
            ignore_file_pattern = kwargs.get('ignore_file_pattern', None)
            local_model_dir = snapshot_download(
                model_name_or_path,
                revision,
                user_agent=invoked_by,
                ignore_file_pattern=ignore_file_pattern)
        logger.info(f'initialize model from {local_model_dir}')

        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg = Config.from_file(
                osp.join(local_model_dir, ModelFile.CONFIGURATION))
        task_name = cfg.task
        if 'task' in kwargs:
            task_name = kwargs.pop('task')
        if isinstance(device, str) and device.startswith('gpu'):
            device = 'cuda' + device[3:]

        automodel_class = get_automodel_class(local_model_dir, task_name)
        if automodel_class is not None:
            default_device_map = None
            if isinstance(device, str):
                if device.startswith('cuda'):
                    default_device_map = {'': 'cuda:0'}
                elif device == 'cpu':
                    default_device_map = {'': 'cpu'}
            device_map = kwargs.get('device_map', default_device_map)
            torch_dtype = kwargs.get('torch_dtype', None)
            config = kwargs.get('config', None)

            model = automodel_class.from_pretrained(
                local_model_dir,
                device_map=device_map,
                torch_dtype=torch_dtype,
                config=config,
                trust_remote_code=True)
            return model

        model_cfg = cfg.model
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type
        model_cfg.model_dir = local_model_dir

        # install and import remote repos before build
        register_plugins_repo(cfg.safe_get('plugins'))
        register_modelhub_repo(local_model_dir, cfg.get('allow_remote', False))

        for k, v in kwargs.items():
            model_cfg[k] = v
        if device is not None:
            model_cfg.device = device
        if task_name is Tasks.backbone:
            model_cfg.init_backbone = True
            model = build_backbone(model_cfg)
        else:
            model = build_model(model_cfg, task_name=task_name)

        # dynamically add pipeline info to model for pipeline inference
        if hasattr(cfg, 'pipeline'):
            model.pipeline = cfg.pipeline

        if not hasattr(model, 'cfg'):
            model.cfg = cfg

        model_cfg.pop('model_dir', None)
        model.name = model_name_or_path
        model.model_dir = local_model_dir
        return model

    def save_pretrained(self,
                        target_folder: Union[str, os.PathLike],
                        save_checkpoint_names: Union[str, List[str]] = None,
                        config: Optional[dict] = None,
                        **kwargs):
        """save the pretrained model, its configuration and other related files to a directory,
            so that it can be re-loaded

        Args:
            target_folder (Union[str, os.PathLike]):
            Directory to which to save. Will be created if it doesn't exist.

            save_checkpoint_names (Union[str, List[str]]):
            The checkpoint names to be saved in the target_folder

            config (Optional[dict], optional):
            The config for the configuration.json, might not be identical with model.config
        """
        raise NotImplementedError(
            'save_pretrained method need to be implemented by the subclass.')

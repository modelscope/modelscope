import os.path as osp
from typing import Optional

from modelscope.hub.file_download import model_file_download
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile


class ModelTypeHelper:

    current_model_type = None

    @staticmethod
    def _get_file_name(model: str, cfg_name: str,
                       revision: Optional[str]) -> Optional[str]:
        if osp.exists(model):
            return osp.join(model, cfg_name)
        try:
            return model_file_download(model, cfg_name, revision=revision)
        except Exception:
            return None

    @staticmethod
    def _parse_and_get(file: Optional[str], pattern: str) -> Optional[str]:
        if file is None or not osp.exists(file):
            return None
        return Config.from_file(file).safe_get(pattern)

    @classmethod
    def _get(cls, model: str, revision: Optional[str]) -> Optional[str]:
        cfg_file = cls._get_file_name(model, ModelFile.CONFIGURATION, revision)
        hf_cfg_file = cls._get_file_name(model, ModelFile.CONFIG, revision)
        cfg_model_type = cls._parse_and_get(cfg_file, 'model.type')
        hf_cfg_model_type = cls._parse_and_get(hf_cfg_file, 'model_type')
        return cfg_model_type or hf_cfg_model_type

    @classmethod
    def _get_adapter(cls, model: str,
                     revision: Optional[str]) -> Optional[str]:
        cfg_file = cls._get_file_name(model, ModelFile.CONFIGURATION, revision)
        model = cls._parse_and_get(cfg_file, 'adapter_cfg.model_id_or_path')
        revision = cls._parse_and_get(cfg_file, 'adapter_cfg.model_revision')
        return None if model is None else cls._get(model, revision)

    @classmethod
    def get(cls,
            model: str,
            revision: Optional[str] = None,
            with_adapter: bool = False,
            split: Optional[str] = None,
            use_cache: bool = False) -> Optional[str]:
        if use_cache and cls.current_model_type:
            return cls.current_model_type
        model_type = cls._get(model, revision)
        if model_type is None and with_adapter:
            model_type = cls._get_adapter(model, revision)
        if model_type is None:
            return None
        model_type = model_type.lower()
        if split is not None:
            model_type = model_type.split(split)[0]
        if use_cache:
            cls.current_model_type = model_type
        return model_type

    @classmethod
    def clear_cache(cls):
        cls.current_model_type = None

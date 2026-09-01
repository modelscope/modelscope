# Copyright (c) Alibaba, Inc. and its affiliates.
"""受限的模型架构配置校验与实例化工具。"""

from __future__ import annotations
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class ArchitectureConfigError(ValueError):
    """模型配置未满足受限架构契约。"""


def require_trust_remote_code(trust_remote_code: bool, context: str) -> None:
    """在读取模型仓库中可影响执行的文件前强制执行授权门禁。"""
    if not trust_remote_code:
        raise RuntimeError(
            f'{context} requires `trust_remote_code=True` because it loads '
            'an architecture or artifact defined by the model repository.')


def validate_mapping_schema(
    value: Any,
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> Mapping[str, Any]:
    """验证 mapping 的字段集合，拒绝意外字段。"""
    if not isinstance(value, Mapping):
        raise ArchitectureConfigError(f'{context} must be a mapping.')
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        missing_fields = ', '.join(sorted(missing))
        raise ArchitectureConfigError(
            f'{context} is missing required fields: {missing_fields}.')
    if unknown:
        unknown_fields = ', '.join(sorted(unknown))
        raise ArchitectureConfigError(
            f'{context} contains unsupported fields: {unknown_fields}.')
    return value


def _validate_data(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArchitectureConfigError(
                    f'{context} contains a non-string key.')
            _validate_data(child, f'{context}.{key}')
        return
    if isinstance(value, Sequence) and not isinstance(value,
                                                      (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_data(child, f'{context}[{index}]')
        return
    raise ArchitectureConfigError(
        f'{context} contains an unsupported value of type {type(value).__name__}.'
    )


def _validate_constructor_arguments(factory: Callable[..., Any],
                                    params: Mapping[str, Any],
                                    context: str) -> None:
    signature = inspect.signature(factory)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD
           for parameter in signature.parameters.values()):
        return
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if name != 'self' and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    unknown = set(params) - allowed
    if unknown:
        unknown_fields = ', '.join(sorted(unknown))
        raise ArchitectureConfigError(
            f'{context}.params contains unsupported fields: {unknown_fields}.')


def instantiate_registered_architecture(
        config: Any,
        registry: Mapping[str, Callable[[], Callable[..., Any]]],
        *,
        context: str,
        sentinels: frozenset[str] = frozenset(),
) -> Any:
    """从固定注册表实例化配置中的架构，绝不导入配置指定的模块。"""
    if isinstance(config, str) and config in sentinels:
        return None
    config = validate_mapping_schema(
        config, required={'target'}, optional={'params'}, context=context)
    target = config['target']
    if not isinstance(target, str):
        raise ArchitectureConfigError(f'{context}.target must be a string.')
    factory_loader = registry.get(target)
    if factory_loader is None:
        raise ArchitectureConfigError(
            f'{context}.target {target!r} is not an approved architecture.')

    params = config.get('params', {})
    if not isinstance(params, Mapping):
        raise ArchitectureConfigError(f'{context}.params must be a mapping.')
    _validate_data(params, f'{context}.params')

    factory = factory_loader()
    _validate_constructor_arguments(factory, params, context)
    return factory(**dict(params))

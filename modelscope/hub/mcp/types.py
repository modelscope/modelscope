# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from typing import Any, Dict, Optional, TypedDict


class McpFilter(TypedDict, total=False):
    """MCP 服务过滤条件类型定义"""
    category: str  # 字符串类型，按类别过滤
    is_hosted: bool  # 布尔类型，按是否托管过滤
    tag: str  # JSON 字符串类型，按标签过滤


def validate_mcp_filter(filter_dict: Optional[Dict[str, Any]]) -> McpFilter:
    """
    验证 MCP filter 参数
    
    Args:
        filter_dict: 待验证的过滤条件字典
        
    Returns:
        McpFilter: 验证后的过滤条件
        
    Raises:
        ValueError: 当参数类型不正确时
    """
    if filter_dict is None:
        return {}
    
    if not isinstance(filter_dict, dict):
        raise ValueError("filter 参数必须是字典类型")
    
    validated_filter = {}
    
    # 验证 category 参数
    if 'category' in filter_dict:
        if not isinstance(filter_dict['category'], str):
            raise ValueError("filter['category'] 必须是字符串类型")
        validated_filter['category'] = filter_dict['category']
    
    # 验证 is_hosted 参数
    if 'is_hosted' in filter_dict:
        if not isinstance(filter_dict['is_hosted'], bool):
            raise ValueError("filter['is_hosted'] 必须是布尔类型")
        validated_filter['is_hosted'] = filter_dict['is_hosted']
    
    # 验证 tag 参数
    if 'tag' in filter_dict:
        if not isinstance(filter_dict['tag'], str):
            raise ValueError("filter['tag'] 必须是字符串类型")
        # 尝试解析 JSON 字符串
        try:
            json.loads(filter_dict['tag'])
            validated_filter['tag'] = filter_dict['tag']
        except json.JSONDecodeError:
            raise ValueError("filter['tag'] 必须是有效的 JSON 字符串")
    
    return validated_filter


def validate_filter_params(func):
    """验证 filter 参数的装饰器"""
    def wrapper(self, token: str, filter: dict = None, *args, **kwargs):
        if filter is not None:
            # 使用 validate_mcp_filter 函数进行验证
            validate_mcp_filter(filter)
        return func(self, token, filter, *args, **kwargs)
    return wrapper 
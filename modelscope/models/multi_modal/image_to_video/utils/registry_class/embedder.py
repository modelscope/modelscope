# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from ..registry import Registry, build_from_config

def build_embedder(cfg, registry, **kwargs):
    """
    Except for ordinal UNet config, if passing a list of dataset config, then return the concat type of it
    """
    return build_from_config(cfg, registry, **kwargs)

EMBEDDER = Registry("EMBEDDER", build_func=build_embedder)

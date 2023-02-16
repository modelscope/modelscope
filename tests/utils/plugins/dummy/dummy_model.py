# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.models.base import Model
from modelscope.models.builder import MODELS


@MODELS.register_module(group_key='dummy-group', module_name='dummy-model')
class DummyModel(Model):
    pass

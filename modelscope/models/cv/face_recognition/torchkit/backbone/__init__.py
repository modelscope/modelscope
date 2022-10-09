# The implementation is adopted from TFace,made pubicly available under the Apache-2.0 license at
# https://github.com/Tencent/TFace/blob/master/recognition/torchkit/backbone
from .model_irse import (IR_18, IR_34, IR_50, IR_101, IR_152, IR_200, IR_SE_50,
                         IR_SE_101, IR_SE_152, IR_SE_200)
from .model_resnet import ResNet_50, ResNet_101, ResNet_152

_model_dict = {
    'ResNet_50': ResNet_50,
    'ResNet_101': ResNet_101,
    'ResNet_152': ResNet_152,
    'IR_18': IR_18,
    'IR_34': IR_34,
    'IR_50': IR_50,
    'IR_101': IR_101,
    'IR_152': IR_152,
    'IR_200': IR_200,
    'IR_SE_50': IR_SE_50,
    'IR_SE_101': IR_SE_101,
    'IR_SE_152': IR_SE_152,
    'IR_SE_200': IR_SE_200
}


def get_model(key):
    """ Get different backbone network by key,
        support ResNet50, ResNet_101, ResNet_152
        IR_18, IR_34, IR_50, IR_101, IR_152, IR_200,
        IR_SE_50, IR_SE_101, IR_SE_152, IR_SE_200.
    """
    if key in _model_dict.keys():
        return _model_dict[key]
    else:
        raise KeyError('not support model {}'.format(key))

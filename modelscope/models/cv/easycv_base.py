# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.base import BaseModel
from easycv.utils.ms_utils import EasyCVMeta

from modelscope.models.base import TorchModel


class EasyCVBaseModel(BaseModel, TorchModel):
    """Base model for EasyCV."""

    def __init__(self, model_dir=None, args=(), kwargs={}):
        kwargs.pop(EasyCVMeta.ARCH, None)  # pop useless keys
        BaseModel.__init__(self)
        TorchModel.__init__(self, model_dir=model_dir)

    def forward(self, img, mode='train', **kwargs):
        if self.training:
            losses = self.forward_train(img, **kwargs)
            loss, log_vars = self._parse_losses(losses)
            return dict(loss=loss, log_vars=log_vars)
        else:
            return self.forward_test(img, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

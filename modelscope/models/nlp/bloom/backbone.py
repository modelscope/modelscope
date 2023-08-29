# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import BloomConfig
from transformers import BloomModel as BloomModelTransform

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import BACKBONES
from modelscope.utils.constant import Tasks


class MsModelMixin:

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.
        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """

        model_dir = kwargs.pop('model_dir', None)
        kwargs.pop('device', None)
        if model_dir is None:
            config = BloomConfig(**kwargs)
            model = cls(config)
        else:
            model = super(MsModelMixin, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **kwargs)
        model.model_dir = model_dir
        return model


@BACKBONES.register_module(group_key=Tasks.backbone, module_name=Models.bloom)
class BloomModel(MsModelMixin, BloomModelTransform, TorchModel):

    pass

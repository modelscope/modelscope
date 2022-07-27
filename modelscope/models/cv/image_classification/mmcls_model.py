import os

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    Tasks.image_classification_imagenet,
    module_name=Models.classification_model)
@MODELS.register_module(
    Tasks.image_classification_dailylife,
    module_name=Models.classification_model)
class ClassificationModel(TorchModel):

    def __init__(self, model_dir: str):
        import mmcv
        from mmcls.models import build_classifier

        super().__init__(model_dir)

        config = os.path.join(model_dir, 'config.py')

        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        self.cls_model = build_classifier(cfg.model)

        self.cfg = cfg
        self.ms_model_dir = model_dir

        self.load_pretrained_checkpoint()

    def forward(self, Inputs):

        return self.cls_model(**Inputs)

    def load_pretrained_checkpoint(self):
        import mmcv
        checkpoint_path = os.path.join(self.ms_model_dir, 'checkpoints.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = mmcv.runner.load_checkpoint(
                self.cls_model, checkpoint_path, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                self.cls_model.CLASSES = checkpoint['meta']['CLASSES']
                self.CLASSES = self.cls_model.CLASSES

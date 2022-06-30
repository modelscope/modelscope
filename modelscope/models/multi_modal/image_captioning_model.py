import os.path as osp
from typing import Any, Dict

import torch.cuda
from PIL import Image

from modelscope.metainfo import Models
from modelscope.utils.constant import ModelFile, Tasks
from ..base import Model
from ..builder import MODELS

__all__ = ['OfaForImageCaptioning']


@MODELS.register_module(Tasks.image_captioning, module_name=Models.ofa)
class OfaForImageCaptioning(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir=model_dir, *args, **kwargs)
        ckpt_name = ModelFile.TORCH_MODEL_FILE
        local_model = osp.join(model_dir, ckpt_name)
        bpe_dir = model_dir
        # turn on cuda if GPU is available
        from fairseq import checkpoint_utils, tasks, utils
        from ofa.tasks.mm_tasks import CaptionTask
        from ofa.utils.eval_utils import eval_caption
        self.eval_caption = eval_caption

        tasks.register_task('caption', CaptionTask)
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.use_fp16 = kwargs[
            'use_fp16'] if 'use_fp16' in kwargs and torch.cuda.is_available()\
            else False
        overrides = {
            'bpe_dir': bpe_dir,
            'eval_cider': False,
            'beam': 5,
            'max_len_b': 16,
            'no_repeat_ngram_size': 3,
            'seed': 7
        }
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(local_model), arg_overrides=overrides)
        # Move models to GPU
        for model in models:
            model.eval()
            model.to(self._device)
            if self.use_fp16:
                model.half()
            model.prepare_for_inference_(cfg)
        self.models = models
        # Initialize generator
        self.generator = task.build_generator(models, cfg.generation)

        # Initialize transform
        from torchvision import transforms
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize(
                (cfg.task.patch_image_size, cfg.task.patch_image_size),
                interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.task = task

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        import fairseq.utils
        if torch.cuda.is_available():
            input = fairseq.utils.move_to_cuda(input, device=self._device)
        results, _ = self.eval_caption(self.task, self.generator, self.models,
                                       input)
        return {
            'image_id': results[0]['image_id'],
            'caption': results[0]['caption']
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # What should we do here ?
        return inputs

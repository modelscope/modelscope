from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from modelscope.pipelines.base import Input
from modelscope.preprocessors import load_image
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ..base import Pipeline
from ..builder import PIPELINES

logger = get_logger()


@PIPELINES.register_module(Tasks.image_captioning, module_name='ofa')
class ImageCaptionPipeline(Pipeline):
    # TODO: refine using modelhub
    def __init__(self, model: str, bpe_dir: str):
        super().__init__()
        # turn on cuda if GPU is available
        from fairseq import checkpoint_utils, tasks, utils
        from ofa.tasks.mm_tasks import CaptionTask

        tasks.register_task('caption', CaptionTask)
        use_cuda = False
        # use fp16 only when GPU is available
        use_fp16 = False
        overrides = {
            'bpe_dir': bpe_dir,
            'eval_cider': False,
            'beam': 5,
            'max_len_b': 16,
            'no_repeat_ngram_size': 3,
            'seed': 7
        }
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(model), arg_overrides=overrides)

        # Move models to GPU
        for model in models:
            model.eval()
            if use_cuda:
                model.cuda()
            if use_fp16:
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
        self.bos_item = torch.LongTensor([task.src_dict.bos()])
        self.eos_item = torch.LongTensor([task.src_dict.eos()])
        self.pad_idx = task.src_dict.pad()

    def preprocess(self, input: Input) -> Dict[str, Any]:

        def encode_text(text, length=None, append_bos=False, append_eos=False):
            s = self.task.tgt_dict.encode_line(
                line=self.task.bpe.encode(text),
                add_if_not_exist=False,
                append_eos=False).long()
            if length is not None:
                s = s[:length]
            if append_bos:
                s = torch.cat([self.bos_item, s])
            if append_eos:
                s = torch.cat([s, self.eos_item])
            return s

        patch_image = self.patch_resize_transform(
            load_image(input)).unsqueeze(0)
        patch_mask = torch.tensor([True])
        text = 'what does the image describe?'
        src_text = encode_text(
            text, append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor(
            [s.ne(self.pad_idx).long().sum() for s in src_text])
        sample = {
            'id': np.array(['42']),
            'net_input': {
                'src_tokens': src_text,
                'src_lengths': src_length,
                'patch_images': patch_image,
                'patch_masks': patch_mask,
            }
        }
        return sample

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        from ofa.utils.eval_utils import eval_caption

        results, _ = eval_caption(self.task, self.generator, self.models,
                                  input)
        return {
            'image_id': results[0]['image_id'],
            'caption': results[0]['caption']
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # What should we do here ?
        return inputs

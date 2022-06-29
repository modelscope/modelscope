# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, Union

import numpy as np
import torch
from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.utils.constant import Fields, ModelFile
from modelscope.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS
from .image import load_image

__all__ = [
    'OfaImageCaptionPreprocessor',
    'MPlugVisualQuestionAnsweringPreprocessor',
]


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.ofa_image_caption)
class OfaImageCaptionPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

        if osp.exists(model_dir):
            local_model_dir = model_dir
        else:
            local_model_dir = snapshot_download(model_dir)
        local_model = osp.join(local_model_dir, ModelFile.TORCH_MODEL_FILE)
        bpe_dir = local_model_dir

        from fairseq import checkpoint_utils, tasks, utils
        from ofa.tasks.mm_tasks import CaptionTask

        tasks.register_task('caption', CaptionTask)

        overrides = {
            'bpe_dir': bpe_dir,
            'eval_cider': False,
            'beam': 5,
            'max_len_b': 16,
            'no_repeat_ngram_size': 3,
            'seed': 7
        }
        model, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(local_model), arg_overrides=overrides)
        del model
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

    @type_assert(object, (str, tuple, Image.Image))
    def __call__(self, data: Union[str, tuple]) -> Dict[str, Any]:

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

        if isinstance(data, Image.Image):
            patch_image = self.patch_resize_transform(data).unsqueeze(0)
        else:
            patch_image = self.patch_resize_transform(
                load_image(data)).unsqueeze(0)
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


@PREPROCESSORS.register_module(
    Fields.multi_modal,
    module_name=Preprocessors.mplug_visual_question_answering)
class MPlugVisualQuestionAnsweringPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via 'bert-base-uncased' tokenizer and configuration

        """
        super().__init__(*args, **kwargs)

        # tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # load configuration
        from sofa.models.mplug import CONFIG_NAME, MPlugConfig
        config = MPlugConfig.from_yaml_file(osp.join(model_dir, CONFIG_NAME))

        # Initialize transform
        from torchvision import transforms
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        self.patch_resize_transform = transforms.Compose([
            transforms.Resize((config.image_res, config.image_res),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, question = data['image'], data['question']
        image = Image.open(image).convert('RGB') if isinstance(image,
                                                               str) else image
        image = self.patch_resize_transform(image)
        image = torch.stack([image], dim=0)
        question = self.tokenizer([question.lower()],
                                  padding='longest',
                                  return_tensors='pt')

        return {'image': image, 'question': question, 'train': False}

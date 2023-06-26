# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

import torch
from collections.abc import Mapping
from torch.utils.data import Dataset
import shutil
from diffusers import DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.outputs import ModelOutputBase
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import \
    CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict

class DreamboothCheckpointProcessor(CheckpointProcessor):

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None):
        """Save the state dict for dreambooth model.
        """
        pipeline_args = {}
        if trainer.model.text_encoder is not None:
            pipeline_args["text_encoder"] = trainer.model.text_encoder
        pipeline = DiffusionPipeline.from_pretrained(
            './dreambooth_diffusion_model',
            unet=trainer.model.unet,
            **pipeline_args,
        )
        scheduler_args = {}
        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(output_dir)  


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


@TRAINERS.register_module(module_name=Trainers.dreambooth_diffusion)
class DreamboothDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_prior_preservation = kwargs.pop('with_prior_preservation', False)
        self.instance_prompt = kwargs.pop('instance_prompt', "a photo of sks dog")
        self.class_data_dir = kwargs.pop("class_data_dir", "/tmp/class_data")
        self.num_class_images = kwargs.pop("num_class_images", 200)

        # save checkpoint and configurate files.
        ckpt_hook = list(filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks))[0]
        ckpt_hook.set_processor(DreamboothCheckpointProcessor())

        # Generate class images if prior preservation is enabled.
        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    revision=self.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images - cur_class_images
                sample_dataset = PromptDataset(self.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.sample_batch_size)

                pipeline.to(self.device)

                for example in tqdm(sample_dataloader, desc="Generating class images"):
                    images = pipeline(example["prompt"]).images
                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.model.unet.parameters(),
                cfg=cfg,
                default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import hashlib
from modelscope.outputs import OutputKeys
from modelscope.utils.torch_utils import is_master
from modelscope.pipelines import pipeline


class DreamBoothDataset(Dataset):

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_data_root=None,
        class_prompt=None,
        with_prior_preservation=False,
        model=None,
    ):
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.with_prior_preservation = with_prior_preservation
        self.model = model
        self._prepared = False

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def prepare_dataset(self):
        if not self._prepared and self.with_prior_preservation and is_master():
            if not self.class_data_root.exists():
                self.class_data_root.mkdir(parents=True)
            cur_class_images = len(list(self.class_data_root.iterdir()))

            if cur_class_images < self.num_class_images:
                pipeline_ins = pipeline('image_generation', model=self.model)
                num_new_images = self.num_class_images - cur_class_images
                for index in range(num_new_images):
                    image = pipeline_ins(self.class_prompt).image
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                            self.class_data_root
                            / f"{index + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

                del pipeline_ins
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._prepared = True

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        self.prepare_dataset()
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        example['instance_images'] = instance_image
        example["instance_prompt_ids"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            example['class_images'] = class_image
            example["class_prompt_ids"] = self.class_prompt
        return example


def initialize_dream_booth(model,
                           instance_data_root,
                           instance_prompt,
                           class_data_root=None,
                           class_prompt=None,
                           with_prior_preservation=False):
    dataset = DreamBoothDataset(instance_data_root, instance_prompt,
                                class_data_root, class_prompt,
                                with_prior_preservation,
                                model=model)
    return dataset


def add_dream_booth_hook(trainer, prior_loss_weight: float):
    from modelscope.trainers.hooks import Hook

    class DreamBoothHook(Hook):

        PRIORITY = 5

        def after_train_iter(self, trainer):
            model_pred, model_pred_prior = torch.chunk(trainer.train_outputs[OutputKeys.LOGITS], 2, dim=0)
            target, target_prior = torch.chunk(trainer.train_outputs['target'], 2, dim=0)

            # Compute instance loss
            loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
                .mean()
            )

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + prior_loss_weight * prior_loss
            trainer.train_outputs['loss'] = loss

        def strategy(self):
            Hook.overload(self.save_checkpoints, name='CheckpointHook.save_checkpoints')
            Hook.overload(self.remove_checkpoints, name='CheckpointHook.remove_checkpoints')
            Hook.overload(self.load_checkpoints, name='CheckpointHook.load_checkpoints')

        def save_checkpoints(self):
            pass

        def load_checkpoints(self):
            pass

        def remove_checkpoints(self):
            pass

    trainer.register_hook(DreamBoothHook())

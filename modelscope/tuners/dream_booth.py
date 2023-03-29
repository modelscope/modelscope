import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.module import register_module_forward_hook
from torch.utils.data import Dataset
from torchvision import transforms

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.torch_utils import is_master
from .base_tuner import Tuner


class DreamBoothDataset(Dataset):

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_data_root=None,
        class_prompt=None,
        with_prior_preservation=False,
        model=None,
        preprocess_data=True,
        tokenizer=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
        **kwargs,
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
        self.preprocess_data = preprocess_data
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        img_transforms = []

        if resize:
            img_transforms.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR))
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose([
            *img_transforms,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

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
                        / f'{index + cur_class_images}-{hash_image}.jpg')
                    image.save(image_filename)

                del pipeline_ins
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._prepared = True

    def preprocess(self, item):
        instance_image = item['instance_images']
        instance_prompt = item['instance_prompt_ids']
        if not instance_image.mode == 'RGB':
            instance_image = instance_image.convert('RGB')
        item['instance_images'] = self.image_transforms(instance_image)
        item['instance_prompt_ids'] = self.tokenizer(
            instance_prompt,
            padding='do_not_pad',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = item['class_images']
            class_prompt = item['class_prompt_ids']
            if not class_image.mode == 'RGB':
                class_image = class_image.convert('RGB')
            item['class_images'] = self.image_transforms(class_image)
            item['class_prompt_ids'] = self.tokenizer(
                class_prompt,
                padding='do_not_pad',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return item

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        self.prepare_dataset()
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        example['instance_images'] = instance_image
        example['instance_prompt_ids'] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images])
            example['class_images'] = class_image
            example['class_prompt_ids'] = self.class_prompt
        if self.preprocess_data:
            example = self.preprocess(example)
        return example


class DreamBoothTuner(Tuner):

    def tune(self,
             model,
             tokenizer,
             instance_data_dir,
             instance_prompt,
             class_data_dir=None,
             class_prompt=None,
             with_prior_preservation=False,
             prior_loss_weight=0.9,
             **kwargs):
        dataset = DreamBoothDataset(
            instance_data_root=instance_data_dir,
            instance_prompt=instance_prompt,
            class_data_root=class_data_dir,
            class_prompt=class_prompt,
            with_prior_preservation=with_prior_preservation,
            tokenizer=tokenizer,
            # model=model,
            **kwargs)

        def loss_calculator(model_pred, target):
            if with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(
                    model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = (
                    F.mse_loss(
                        model_pred.float(), target.float(),
                        reduction='none').mean([1, 2, 3]).mean())

                # Compute prior loss
                prior_loss = F.mse_loss(
                    model_pred_prior.float(),
                    target_prior.float(),
                    reduction='mean')

                # Add the prior loss to the instance loss.
                loss = loss + prior_loss_weight * prior_loss
                return loss
            else:
                loss = F.mse_loss(model_pred, target, reduction='mean')
                return loss

        def collate_fn(examples):
            input_ids = [
                example['instance_prompt_ids'] for example in examples
            ]
            pixel_values = [example['instance_images'] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if with_prior_preservation:
                input_ids += [
                    example['class_prompt_ids'] for example in examples
                ]
                pixel_values += [
                    example['class_images'] for example in examples
                ]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format).float()

            input_ids = tokenizer.pad(
                {
                    'input_ids': input_ids
                },
                padding='max_length',
                max_length=tokenizer.model_max_length,
                return_tensors='pt',
            ).input_ids

            batch = {
                'input_ids': input_ids,
                'pixel_values': pixel_values,
            }
            return batch

        # return dataset, collate_fn, loss_calculator
        return dataset, collate_fn, loss_calculator

    def add_hook(self, trainer):
        pass

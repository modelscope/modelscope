# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import hashlib
import itertools
import os
import random
import warnings
from pathlib import Path
from typing import Union

import json
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import \
    CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import ModeKeys, TrainerStages
from modelscope.utils.data_utils import to_device
from modelscope.utils.torch_utils import is_dist


class CustomCheckpointProcessor(CheckpointProcessor):

    def __init__(self,
                 modifier_token,
                 modifier_token_id,
                 torch_type=torch.float32,
                 safe_serialization=False):
        """Checkpoint processor for custom diffusion.

        Args:
            modifier_token: The token to use as a modifier for the concept.
            modifier_token_id: The modifier token id for the concept.
            torch_type: The torch type, default is float32.
            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.
        """
        self.modifier_token = modifier_token
        self.modifier_token_id = modifier_token_id
        self.torch_type = torch_type
        self.safe_serialization = safe_serialization

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        """Save the state dict for custom diffusion model.
        """
        trainer.model.unet = trainer.model.unet.to(self.torch_type)
        trainer.model.unet.save_attn_procs(
            output_dir, safe_serialization=self.safe_serialization)

        learned_embeds = trainer.model.text_encoder.get_input_embeddings(
        ).weight
        if not isinstance(self.modifier_token_id, list):
            self.modifier_token_id = [self.modifier_token_id]
        for x, y in zip(self.modifier_token_id, self.modifier_token):
            learned_embeds_dict = {}
            learned_embeds_dict[y] = learned_embeds[x]
            torch.save(learned_embeds_dict, f'{output_dir}/{y}.bin')


class CustomDiffusionDataset(Dataset):

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        """A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
        It pre-processes the images and the tokenizes prompts.

        Args:
            concepts_list: contain multiple concepts, instance_prompt, class_prompt, etc.
            tokenizer: pretrained tokenizer.
            size: the size of images.
            mask_size: the mask size of images.
            center_crop: execute center crop or not.
            with_prior_preservation: flag to add prior preservation loss.
            hflip: whether to flip horizontally.
            aug: perform data augmentation.

        """
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [
                (x, concept['instance_prompt'])
                for x in Path(concept['instance_data_dir']).iterdir()
                if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept['class_data_dir'])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [
                        concept['class_prompt']
                        for _ in range(len(class_images_path))
                    ]
                else:
                    with open(class_data_root, 'r') as f:
                        class_images_path = f.read().splitlines()
                    with open(concept['class_prompt'], 'r') as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [
                    (x, y) for (x, y) in zip(class_images_path, class_prompt)
                ]
                self.class_images_path.extend(
                    class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose([
            self.flip,
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size)
            if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(
            0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top:top + inner, left:left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top:top + inner, left:left + inner, :] = image
            mask[top // factor + 1:(top + scale) // factor - 1,
                 left // factor + 1:(left + scale) // factor - 1] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[
            index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == 'RGB':
            instance_image = instance_image.convert('RGB')
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size
                                  + 1) if np.random.uniform() < 0.66 else
                np.random.randint(int(1.2 * self.size), int(1.4 * self.size)))
        instance_image, mask = self.preprocess(instance_image, random_scale,
                                               self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(['a far away ', 'very small '
                                                ]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(['zoomed in ', 'close up '
                                                ]) + instance_prompt

        example['instance_images'] = torch.from_numpy(instance_image).permute(
            2, 0, 1)
        example['mask'] = torch.from_numpy(mask)
        example['instance_prompt_ids'] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[
                index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == 'RGB':
                class_image = class_image.convert('RGB')
            example['class_images'] = self.image_transforms(class_image)
            example['class_mask'] = torch.ones_like(example['mask'])
            example['class_prompt_ids'] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
            ).input_ids

        return example


class PromptDataset(Dataset):

    def __init__(self, prompt, num_samples):
        """Dataset to prepare the prompts to generate class images.

        Args:
            prompt: Class prompt.
            num_samples: The number sample for class images.

        """
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example['prompt'] = self.prompt
        example['index'] = index
        return example


@TRAINERS.register_module(module_name=Trainers.custom_diffusion)
class CustomDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Custom diffusion trainers for fine-tuning stable diffusion.

        Args:
            with_prior_preservation: a boolean indicating whether to enable prior loss.
            instance_prompt: a string specifying the instance prompt.
            class_prompt: a string specifying the class prompt.
            class_data_dir: the path to the class data directory.
            num_class_images: the number of class images to generate.
            prior_loss_weight: the weight of the prior loss.
            modifier_token: A token to use as a modifier for the concept.
            initializer_token: A token to use as initializer word.
            freeze_model: crossattn to enable fine-tuning of all params in the cross attention.
            sample_batch_size: Batch size (per device) for sampling images.
            train_batch_size: Batch size (per device) for the training dataloader.
            center_crop: execute center crop or not.
            concepts_list: Path to json containing multiple concepts, will overwrite parameters.
            instance_data_name: The instance data local dir or online ID.
            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.

        """
        self.with_prior_preservation = kwargs.pop('with_prior_preservation',
                                                  True)
        instance_prompt = kwargs.pop('instance_prompt', 'a photo of sks dog')
        class_prompt = kwargs.pop('class_prompt', 'dog')
        class_data_dir = kwargs.pop('class_data_dir', '/tmp/class_data')
        self.torch_type = kwargs.pop('torch_type', torch.float32)
        self.real_prior = kwargs.pop('real_prior', False)
        self.num_class_images = kwargs.pop('num_class_images', 200)
        self.resolution = kwargs.pop('resolution', 512)
        self.prior_loss_weight = kwargs.pop('prior_loss_weight', 1.0)
        self.modifier_token = kwargs.pop('modifier_token', '<new1>')
        self.initializer_token = kwargs.pop('initializer_token', 'ktn+pll+ucd')
        self.freeze_model = kwargs.pop('freeze_model', 'crossattn_kv')
        self.sample_batch_size = kwargs.pop('sample_batch_size', 4)
        self.train_batch_size = kwargs.pop('train_batch_size', 2)
        self.center_crop = kwargs.pop('center_crop', False)
        self.concepts_list = kwargs.pop('concepts_list', None)
        instance_data_name = kwargs.pop(
            'instance_data_name', 'buptwq/lora-stable-diffusion-finetune-dog')
        safe_serialization = kwargs.pop('safe_serialization', False)

        # Extract downloaded image folder
        if self.concepts_list is None:
            if os.path.isdir(instance_data_name):
                instance_data_dir = instance_data_name
            else:
                ds = MsDataset.load(instance_data_name, split='train')
                instance_data_dir = os.path.dirname(
                    next(iter(ds))['Target:FILE'])

        # construct concept list
        if self.concepts_list is None:
            self.concepts_list = [{
                'instance_prompt': instance_prompt,
                'class_prompt': class_prompt,
                'instance_data_dir': instance_data_dir,
                'class_data_dir': class_data_dir,
            }]
        else:
            with open(self.concepts_list, 'r') as f:
                self.concepts_list = json.load(f)

        for concept in self.concepts_list:
            if not os.path.exists(concept['class_data_dir']):
                os.makedirs(concept['class_data_dir'])
            if not os.path.exists(concept['instance_data_dir']):
                raise Exception(
                    f"instance dataset {concept['instance_data_dir']} does not exist."
                )

        # Adding a modifier token which is optimized
        self.modifier_token_id = []
        initializer_token_id = []
        if self.modifier_token is not None:
            self.modifier_token = self.modifier_token.split('+')
            self.initializer_token = self.initializer_token.split('+')
            if len(self.modifier_token) > len(self.initializer_token):
                raise ValueError(
                    'You must specify + separated initializer token for each modifier token.'
                )
            for modifier_token, initializer_token in zip(
                    self.modifier_token,
                    self.initializer_token[:len(self.modifier_token)]):
                # Add the placeholder token in tokenizer
                num_added_tokens = self.model.tokenizer.add_tokens(
                    modifier_token)
                if num_added_tokens == 0:
                    raise ValueError(
                        f'The tokenizer already contains the token {modifier_token}. Please pass a different'
                        ' `modifier_token` that is not already in the tokenizer.'
                    )

                # Convert the initializer_token, placeholder_token to ids
                token_ids = self.model.tokenizer.encode(
                    [initializer_token], add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError(
                        'The initializer token must be a single token.')

                initializer_token_id.append(token_ids[0])
                self.modifier_token_id.append(
                    self.model.tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.model.text_encoder.resize_token_embeddings(
            len(self.model.tokenizer))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.model.text_encoder.resize_token_embeddings(
            len(self.model.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.model.text_encoder.get_input_embeddings(
        ).weight.data
        for x, y in zip(self.modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.model.text_encoder.text_model.encoder.parameters(),
            self.model.text_encoder.text_model.final_layer_norm.parameters(),
            self.model.text_encoder.text_model.embeddings.position_embedding.
            parameters(),
        )
        self.freeze_params(params_to_freeze)

        # Save checkpoint and configurate files
        ckpt_hook = list(
            filter(lambda hook: isinstance(hook, CheckpointHook),
                   self.hooks))[0]
        ckpt_hook.set_processor(
            CustomCheckpointProcessor(self.modifier_token,
                                      self.modifier_token_id, self.torch_type,
                                      safe_serialization))

        # Add new Custom Diffusion weights to the attention layers
        attention_class = CustomDiffusionAttnProcessor
        # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params.
        train_q_out = False if self.freeze_model == 'crossattn_kv' else True
        custom_diffusion_attn_procs = {}

        st = self.model.unet.state_dict()
        for name, _ in self.model.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith(
                'attn1.processor'
            ) else self.model.unet.config.cross_attention_dim
            if name.startswith('mid_block'):
                hidden_size = self.model.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len('up_blocks.')])
                hidden_size = list(
                    reversed(
                        self.model.unet.config.block_out_channels))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                hidden_size = self.model.unet.config.block_out_channels[
                    block_id]
            layer_name = name.split('.processor')[0]
            weights = {
                'to_k_custom_diffusion.weight':
                st[layer_name + '.to_k.weight'],
                'to_v_custom_diffusion.weight':
                st[layer_name + '.to_v.weight'],
            }
            if train_q_out:
                weights['to_q_custom_diffusion.weight'] = st[layer_name
                                                             + '.to_q.weight']
                weights['to_out_custom_diffusion.0.weight'] = st[
                    layer_name + '.to_out.0.weight']
                weights['to_out_custom_diffusion.0.bias'] = st[
                    layer_name + '.to_out.0.bias']
            if cross_attention_dim is not None:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=True,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(self.model.unet.device)
                custom_diffusion_attn_procs[name].load_state_dict(weights)
            else:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=False,
                    train_q_out=False,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
        del st
        self.model.unet.set_attn_processor(custom_diffusion_attn_procs)
        self.custom_diffusion_layers = AttnProcsLayers(
            self.model.unet.attn_processors)

        # Check for conflicts and conflicts
        if self.with_prior_preservation:
            for concept in self.concepts_list:
                if concept['class_data_dir'] is None:
                    raise ValueError(
                        'You must specify a data directory for class images.')
                if concept['class_prompt'] is None:
                    raise ValueError(
                        'You must specify prompt for class images.')
        else:
            for concept in self.concepts_list:
                if concept['class_data_dir'] is not None:
                    warnings.warn(
                        'You need not use --class_data_dir without --with_prior_preservation.'
                    )
                if concept['class_prompt'] is not None:
                    warnings.warn(
                        'You need not use --class_prompt without --with_prior_preservation.'
                    )

        # Generate class images if prior preservation is enabled.
        if self.with_prior_preservation:
            self.generate_image()

        # Dataset and DataLoaders creation:
        train_dataset = CustomDiffusionDataset(
            concepts_list=self.concepts_list,
            tokenizer=self.model.tokenizer,
            with_prior_preservation=self.with_prior_preservation,
            size=self.resolution,
            mask_size=self.model.vae.encode(
                torch.randn(1, 3, self.resolution,
                            self.resolution).to(dtype=self.torch_type).to(
                                self.device)).latent_dist.sample().size()[-1],
            center_crop=self.center_crop,
            num_class_images=self.num_class_images,
            hflip=False,
            aug=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: self.collate_fn(examples),
            num_workers=2,
        )
        self.iter_train_dataloader = itertools.cycle(train_dataloader)

    def freeze_params(self, params):
        for param in params:
            param.requires_grad = False

    def collate_fn(self, examples):
        input_ids = [example['instance_prompt_ids'] for example in examples]
        pixel_values = [example['instance_images'] for example in examples]
        mask = [example['mask'] for example in examples]
        # Concat class and instance examples which avoid doing two forward passes.
        if self.with_prior_preservation:
            input_ids += [example['class_prompt_ids'] for example in examples]
            pixel_values += [example['class_images'] for example in examples]
            mask += [example['class_mask'] for example in examples]

        input_ids = torch.cat(input_ids, dim=0)
        pixel_values = torch.stack(pixel_values)
        mask = torch.stack(mask)
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        mask = mask.to(memory_format=torch.contiguous_format).float()

        batch = {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'mask': mask.unsqueeze(1)
        }
        return batch

    def generate_image(self):
        """ Generate class images if prior preservation is enabled.
        """
        for i, concept in enumerate(self.concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)

            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.model_dir,
                    safety_checker=None,
                    torch_dtype=self.torch_type,
                    revision=None,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images - cur_class_images

                sample_dataset = PromptDataset(concept['class_prompt'],
                                               num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=self.sample_batch_size)

                pipeline.to(self.device)

                for example in tqdm(
                        sample_dataloader,
                        desc='Generating class images',
                        # disable=not accelerator.is_local_main_process,
                ):
                    images = pipeline(example['prompt']).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        save_index = example['index'][i] + cur_class_images
                        image_filename = class_images_dir / f'{save_index}-{hash_image}.jpg'
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                itertools.chain(
                    self.model.text_encoder.get_input_embeddings().parameters(
                    ), self.custom_diffusion_layers.parameters()),
                cfg=cfg,
                default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e

    def train_loop(self, data_loader):
        """ Training loop used by `EpochBasedTrainer.train()`
        """
        self.invoke_hook(TrainerStages.before_run)
        self.model.train()
        for _ in range(self._epoch, self._max_epochs):
            self.invoke_hook(TrainerStages.before_train_epoch)
            for i, data_batch in enumerate(data_loader):
                if i < self.inner_iter:
                    # inner_iter may be read out from the checkpoint file, so skip the trained iters in the epoch.
                    continue
                data_batch = to_device(data_batch, self.device)
                self.data_batch = data_batch
                self._inner_iter = i
                self.invoke_hook(TrainerStages.before_train_iter)
                self.train_step(self.model, data_batch)
                self.invoke_hook(TrainerStages.after_train_iter)
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept to optimize the concept embeddings.
                if self.modifier_token is not None:
                    grads_text_encoder = self.model.text_encoder.get_input_embeddings(
                    ).weight.grad
                    # Get the index for tokens that we want to zero the grads.
                    index_grads_to_zero = torch.arange(
                        len(self.model.tokenizer)) != self.modifier_token_id[0]
                    for i in range(len(self.modifier_token_id[1:])):
                        modifier_flag = torch.arange(
                            len(self.model.tokenizer)
                        ) != self.modifier_token_id[i]
                        index_grads_to_zero = index_grads_to_zero & modifier_flag
                    grads_data = grads_text_encoder.data[
                        index_grads_to_zero, :].fill_(0)
                    grads_text_encoder.data[
                        index_grads_to_zero, :] = grads_data
                # Value changed after the hooks are invoked, do not move them above the invoke_hook code.
                del self.data_batch
                self._iter += 1
                self._mode = ModeKeys.TRAIN

                if i + 1 >= self.iters_per_epoch:
                    break

            self.invoke_hook(TrainerStages.after_train_epoch)
            # Value changed after the hooks are invoked, do not move them above the invoke_hook code.
            self._inner_iter = 0
            self._epoch += 1
            if self._stop_training:
                break

        self.invoke_hook(TrainerStages.after_run)

    def train_step(self, model, inputs):
        """ Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`TorchModel`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        self.model.unet.train()
        if self.modifier_token is not None:
            self.model.text_encoder.train()
        self._mode = ModeKeys.TRAIN

        batch = next(self.iter_train_dataloader)
        # Convert images to latent space
        latents = self.model.vae.encode(batch['pixel_values'].to(
            dtype=self.torch_type).to(self.device)).latent_dist.sample()
        latents = latents * self.model.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.model.noise_scheduler.config.num_train_timesteps, (bsz, ),
            device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.model.noise_scheduler.add_noise(
            latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.model.text_encoder(batch['input_ids'].to(
            self.device))[0]

        # Predict the noise residual
        model_pred = self.model.unet(noisy_latents, timesteps,
                                     encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.model.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.model.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.model.noise_scheduler.get_velocity(
                latents, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.model.noise_scheduler.config.prediction_type}'
            )

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            mask = torch.chunk(batch['mask'].to(self.device), 2, dim=0)[0]
            # Compute instance loss
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(),
                target_prior.float(),
                reduction='mean')

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            mask = batch['mask'].to(self.device)
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

        train_outputs = {}
        train_outputs[OutputKeys.LOSS] = loss

        # add model output info to log
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update(
                    [key for key in train_outputs.keys() if key_p in key])

            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if is_dist():
                        value = value.data.clone().to('cuda')
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])

        self.train_outputs = train_outputs

# Copyright 2023 The HuggingFace Team.
# Copyright 2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.

# The implementation here is modified based on diffusers,
# originally Apache License, Copyright 2023 The HuggingFace Team

import math
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import \
    StableDiffusionPipelineOutput
from PIL import Image
from tqdm.auto import tqdm

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis, module_name=Pipelines.cones2_inference)
class Cones2InferencePipeline(DiffusersPipeline):
    r""" Cones2 Inference Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> pipeline =pipeline(task=Tasks.text_to_image_synthesis, model= 'damo/Cones2', model_revision='v1.0.1')
    >>>   {
    >>>    "text": 'a mug and a dog on the beach',
    >>>    "subject_list": [["mug", 2], ["dog", 5]],
    >>>    "color_context": {"255,192,0": ["mug", 2.5], "255,0,0": ["dog", 2.5]},
    >>>    "layout": 'data/test/images/mask_example.png'
    >>>   }
    >>>
    """

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub.
            device: str = 'gpu'
        """
        super().__init__(model, device, **kwargs)
        self.pipeline = StableDiffusionPipeline.from_pretrained(model)
        self.pipeline.text_encoder.pooler = None
        self.pipeline.to(self.device)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')

        return self.layout_guidance_sampling(
            prompt=inputs.get('text'),
            residual_dict=inputs.get('residual_dict', None),
            subject_list=inputs.get('subject_list'),
            color_context=inputs.get('color_context', None),
            layout=inputs.get('layout', None),
        )

    @torch.no_grad()
    def layout_guidance_sampling(
        self,
        prompt='',
        residual_dict=None,
        subject_list=None,
        color_context=None,
        layout=None,
        cfg_scale=7.5,
        inference_steps=50,
        guidance_steps=50,
        guidance_weight=0.05,
        weight_negative=-1e8,
    ):

        layout = Image.open(layout).resize((768, 768)).convert('RGB')
        subject_color_dict = {
            tuple(map(int, key.split(','))): value
            for key, value in color_context.items()
        }

        vae = self.pipeline.vae
        unet = self.pipeline.unet
        text_encoder = self.pipeline.text_encoder
        tokenizer = self.pipeline.tokenizer
        unconditional_input_prompt = ''
        scheduler = LMSDiscreteScheduler.from_config(
            self.pipeline.scheduler.config)
        scheduler.set_timesteps(inference_steps, device=self.device)
        if guidance_steps > 0:
            guidance_steps = min(guidance_steps, inference_steps)
            scheduler_guidance = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule='scaled_linear',
                num_train_timesteps=1000,
            )
            scheduler_guidance.set_timesteps(
                guidance_steps, device=self.device)

        # Process input prompt text
        text_input = tokenizer(
            [prompt],
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )

        # Edit text embedding conditions with residual token embeddings.
        cond_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        if residual_dict is not None:
            for name, token in subject_list:
                residual_token_embedding = torch.load(residual_dict[name])
                cond_embeddings[0][token] += residual_token_embedding.reshape(
                    1024)

        # Process unconditional input "" for classifier-free guidance.
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([unconditional_input_prompt],
                                 padding='max_length',
                                 max_length=max_length,
                                 return_tensors='pt')
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(self.device))[0]

        register_attention_control(unet)

        # Calculate the hidden features for each cross attention layer.
        hidden_states, uncond_hidden_states = _extract_cross_attention(
            tokenizer, self.device, layout, subject_color_dict, text_input,
            weight_negative)
        hidden_states['CONDITION_TENSOR'] = cond_embeddings
        uncond_hidden_states['CONDITION_TENSOR'] = uncond_embeddings
        hidden_states['function'] = lambda w, sigma, qk: (
            guidance_weight * w * math.log(1 + sigma**2)) * qk.std()
        uncond_hidden_states['function'] = lambda w, sigma, qk: 0.0

        # Sampling the initial latents.
        latent_size = (1, unet.in_channels, 96, 96)
        latents = torch.randn(latent_size).to(self.device)
        latents = latents * scheduler.init_noise_sigma

        for i, t in tqdm(
                enumerate(scheduler.timesteps),
                total=len(scheduler.timesteps)):
            # Improve the harmony of generated images by self-recurrence.
            if i < guidance_steps:
                loop = 2
            else:
                loop = 1
            for k in range(loop):
                if i < guidance_steps:
                    sigma = scheduler_guidance.sigmas[i]
                    latent_model_input = scheduler.scale_model_input(
                        latents, t)
                    _t = t

                    hidden_states.update({'SIGMA': sigma})

                    noise_pred_text = unet(
                        latent_model_input,
                        _t,
                        encoder_hidden_states=hidden_states,
                    ).sample

                    uncond_hidden_states.update({'SIGMA': sigma})

                    noise_pred_uncond = unet(
                        latent_model_input,
                        _t,
                        encoder_hidden_states=uncond_hidden_states,
                    ).sample

                    noise_pred = noise_pred_uncond + cfg_scale * (
                        noise_pred_text - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents,
                                             1).prev_sample

                    # Self-recurrence.
                    if k < 1 and loop > 1:
                        noise_recurent = torch.randn(latents.shape).to(
                            self.device)
                        sigma_difference = scheduler.sigmas[
                            i]**2 - scheduler.sigmas[i + 1]**2
                        latents = latents + noise_recurent * (
                            sigma_difference**0.5)
                else:
                    latent_model_input = scheduler.scale_model_input(
                        latents, t)
                    _t = t
                    noise_pred_text = unet(
                        latent_model_input,
                        _t,
                        encoder_hidden_states=cond_embeddings,
                    ).sample

                    latent_model_input = scheduler.scale_model_input(
                        latents, t)

                    noise_pred_uncond = unet(
                        latent_model_input,
                        _t,
                        encoder_hidden_states=uncond_embeddings,
                    ).sample

                    noise_pred = noise_pred_uncond + cfg_scale * (
                        noise_pred_text - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents,
                                             1).prev_sample

        edited_images = _latents_to_images(vae, latents)

        return StableDiffusionPipelineOutput(
            images=edited_images, nsfw_content_detected=None)

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}


class Cones2AttnProcessor:

    def __init__(self):
        super().__init__()

    def __call__(self,
                 attn: Attention,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        is_dict_format = True
        if encoder_hidden_states is not None:
            if 'CONDITION_TENSOR' in encoder_hidden_states:
                encoder_hidden = encoder_hidden_states['CONDITION_TENSOR']
            else:
                encoder_hidden = encoder_hidden_states
                is_dict_format = False
        else:
            encoder_hidden = hidden_states

        key = attn.to_k(encoder_hidden)
        value = attn.to_v(encoder_hidden)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_size_of_img = attention_scores.size()[-2]

        if attention_scores.size()[2] == 77:
            if is_dict_format:
                f = encoder_hidden_states['function']
                try:
                    w = encoder_hidden_states[
                        f'CA_WEIGHT_{attention_size_of_img}']
                except KeyError:
                    w = encoder_hidden_states['CA_WEIGHT_ORIG']
                    if not isinstance(w, int):
                        img_h, img_w, nc = w.shape
                        ratio = math.sqrt(img_h * img_w
                                          / attention_size_of_img)
                        w = F.interpolate(
                            w.permute(2, 0, 1).unsqueeze(0),
                            scale_factor=1 / ratio,
                            mode='bilinear',
                            align_corners=True)
                        w = F.interpolate(
                            w.reshape(1, nc, -1),
                            size=(attention_size_of_img, ),
                            mode='nearest').permute(2, 1, 0).squeeze()
                    else:
                        w = 0
                if type(w) is int and w == 0:
                    sigma = encoder_hidden_states['SIGMA']
                    cross_attention_weight = f(w, sigma, attention_scores)
                else:
                    bias = torch.zeros_like(w)
                    bias[torch.where(w > 0)] = attention_scores.std() * 0
                    sigma = encoder_hidden_states['SIGMA']
                    cross_attention_weight = f(w, sigma, attention_scores)
                    cross_attention_weight = cross_attention_weight + bias
            else:
                cross_attention_weight = 0.0
        else:
            cross_attention_weight = 0.0

        attention_scores = (attention_scores
                            + cross_attention_weight) * attn.scale
        attention_probs = attention_scores.softmax(dim=-1)

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(unet):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        attn_procs[name] = Cones2AttnProcessor()

    unet.set_attn_processor(attn_procs)


def _tokens_img_attention_weight(img_context_seperated,
                                 tokenized_texts,
                                 ratio: int = 8,
                                 original_shape=False):
    token_lis = tokenized_texts['input_ids'][0].tolist()
    w, h = img_context_seperated[0][1].shape

    w_r, h_r = round(w / ratio), round(h / ratio)
    ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)
    for v_as_tokens, img_where_color in img_context_seperated:

        is_in = 0

        for idx, tok in enumerate(token_lis):
            if token_lis[idx:idx + len(v_as_tokens)] == v_as_tokens:
                is_in = 1

                ret_tensor[:, idx:idx + len(v_as_tokens)] += (
                    _downsampling(img_where_color, w_r,
                                  h_r).reshape(-1,
                                               1).repeat(1, len(v_as_tokens)))

        if not is_in == 1:
            print(
                f'Warning ratio {ratio} : tokens {v_as_tokens} not found in text'
            )

    if original_shape:
        ret_tensor = ret_tensor.reshape((w_r, h_r, len(token_lis)))

    return ret_tensor


def _image_context_seperator(img, color_context: dict, _tokenizer, neg: float):
    ret_lists = []
    if img is not None:
        w, h = img.size
        matrix = np.zeros((h, w))
        for color, v in color_context.items():
            color = tuple(color)
            if len(color) > 3:
                color = color[:3]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))
            img_where_color = (np.array(img) == color).all(axis=-1)
            matrix[img_where_color] = 1

        for color, (subject, weight_active) in color_context.items():
            if len(color) > 3:
                color = color[:3]
            v_input = _tokenizer(
                subject,
                max_length=_tokenizer.model_max_length,
                truncation=True,
            )

            v_as_tokens = v_input['input_ids'][1:-1]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))
            img_where_color = (np.array(img) == color).all(axis=-1)
            matrix[img_where_color] = 1
            if not img_where_color.sum() > 0:
                print(
                    f'Warning : not a single color {color} not found in image')

            img_where_color_init = torch.where(
                torch.tensor(img_where_color, dtype=torch.bool), weight_active,
                neg)

            img_where_color = torch.where(
                torch.from_numpy(matrix == 1) & (img_where_color_init == 0.0),
                torch.tensor(neg), img_where_color_init)

            ret_lists.append((v_as_tokens, img_where_color))
    else:
        w, h = 768, 768

    if len(ret_lists) == 0:
        ret_lists.append(([-1], torch.zeros((w, h), dtype=torch.float32)))
    return ret_lists, w, h


def _extract_cross_attention(tokenizer, device, color_map_image, color_context,
                             text_input, neg):
    # Process color map image and context
    seperated_word_contexts, width, height = _image_context_seperator(
        color_map_image, color_context, tokenizer, neg)

    # Compute cross-attention weights
    cross_attention_weight_1 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=1,
        original_shape=True).to(device)
    cross_attention_weight_8 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=8).to(device)
    cross_attention_weight_16 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=16).to(device)
    cross_attention_weight_32 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=32).to(device)
    cross_attention_weight_64 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=64).to(device)

    hidden_states = {
        'CA_WEIGHT_ORIG': cross_attention_weight_1,  # 768 x 768
        'CA_WEIGHT_9216': cross_attention_weight_8,  # 96 x 96
        'CA_WEIGHT_2304': cross_attention_weight_16,  # 48 x 48
        'CA_WEIGHT_576': cross_attention_weight_32,  # 24 x 24
        'CA_WEIGHT_144': cross_attention_weight_64,  # 12 x 12
    }

    uncond_hidden_states = {
        'CA_WEIGHT_ORIG': 0,
        'CA_WEIGHT_9216': 0,
        'CA_WEIGHT_2304': 0,
        'CA_WEIGHT_576': 0,
        'CA_WEIGHT_144': 0,
    }

    return hidden_states, uncond_hidden_states


def _downsampling(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        size=(w, h),
        mode='bilinear',
        align_corners=True,
    ).squeeze()


def _latents_to_images(vae, latents, scale_factor=0.18215):
    """Decode latents to PIL images."""
    scaled_latents = 1.0 / scale_factor * latents.clone()
    images = vae.decode(scaled_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def _sanitize_parameters(self, **pipeline_parameters):
    """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method

        Returns:
            Dict[str, str]:  preprocess_params = {'image_resolution': self.model.get_resolution()}
            Dict[str, str]:  forward_params = pipeline_parameters
            Dict[str, str]:  postprocess_params = {}
        """
    pipeline_parameters['image_resolution'] = self.model.get_resolution()
    pipeline_parameters['modelsetting'] = self.model.get_config()
    pipeline_parameters['model_dir'] = self.model.get_model_dir()
    pipeline_parameters['control_type'] = self.init_control_type
    pipeline_parameters['device'] = self.device

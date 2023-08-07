# Copyright © Alibaba, Inc. and its affiliates.
# The implementation here is modifed based on diffusers.StableDiffusionControlNetImg2ImgPipeline,
# originally Apache 2.0 License and public available at
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py

import copy
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DiffusionPipeline,
                       StableDiffusionControlNetImg2ImgPipeline,
                       UNet2DConditionModel)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import ControlNetModel
from diffusers.models.vae import DecoderOutput
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (PIL_INTERPOLATION, deprecate,
                             is_accelerate_available, is_accelerate_version,
                             is_compiled_module, logging, randn_tensor,
                             replace_example_docstring)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> modelscope.models.cv.text_to_360panorama_image import StableDiffusionControlNetImg2ImgPanoPipeline
        >>> base_model_path = "damo/cv_diffusion_text-to-360panorama-image_generation/sr-base"
        >>> controlnet_path = "damo/cv_diffusion_text-to-360panorama-image_generation/sr-control"
        >>> controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetImg2ImgPanoPipeline.from_pretrained(base_model_path, controlnet=controlnet,
        ...                                                                     torch_dtype=torch.float16)
        >>> pipe.vae.enable_tiling()
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()
        >>> pipe.enable_model_cpu_offload()
        >>> input_image_path = 'data/test/images/test_to_360panorama_image/test.png'
        >>> image = Image.open(input_image_path)
        >>> image = pipe(
        ...     "futuristic-looking woman",
        ...     num_inference_steps=20,
        ...     image=image,
        ...     height=768,
        ...     width=1536,
        ...     control_image=image,
        ... ).images[0]

        ```
"""

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [['', 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe: DiffusionPipeline, prompt: List[str],
                             max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning(
            'Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples'
        )
    return tokens, weights


def pad_tokens_and_weights(tokens,
                           weights,
                           max_length,
                           bos,
                           eos,
                           pad,
                           no_boseos_middle=True,
                           chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [
            bos
        ] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (
                max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2):min(
                        len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: DiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2):(i + 1)
                                          * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(text_input)[0]
    return text_embeddings


def get_weighted_text_embeddings(
    pipe: DiffusionPipeline,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length
                  - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(
            pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(
                pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in pipe.tokenizer(
                prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(
                    uncond_prompt, max_length=max_length,
                    truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length,
                         max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length
                  - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = getattr(pipe.tokenizer, 'pad_token_id', eos)
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(
        prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(
            uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(
        prompt_weights,
        dtype=text_embeddings.dtype,
        device=text_embeddings.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(
            uncond_weights,
            dtype=uncond_embeddings.dtype,
            device=uncond_embeddings.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(
            text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(
            text_embeddings.dtype)
        text_embeddings *= (previous_mean
                            / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(
                uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(
                uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean
                                  / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert('RGB'))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


class StableDiffusionControlNetImg2ImgPanoPipeline(
        StableDiffusionControlNetImg2ImgPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/
            model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ['safety_checker', 'feature_extractor']

    def check_inputs(
        self,
        prompt,
        image,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
        condition_1 = callback_steps is not None
        condition_2 = not isinstance(callback_steps,
                                     int) or callback_steps <= 0
        if (callback_steps is None) or (condition_1 and condition_2):
            raise ValueError(
                f'`callback_steps` has to be a positive integer but is {callback_steps} of type'
                f' {type(callback_steps)}.')
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to'
                ' only forward one of the two.')
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
        elif prompt is not None and (not isinstance(prompt, str)
                                     and not isinstance(prompt, list)):
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:'
                f' {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    '`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but'
                    f' got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`'
                    f' {negative_prompt_embeds.shape}.')
        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f'You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}'
                    ' prompts. The conditionings will be fixed across the prompts.'
                )
        # Check `image`
        is_compiled = hasattr(
            F, 'scaled_dot_product_attention') and isinstance(
                self.controlnet, torch._dynamo.eval_frame.OptimizedModule)
        if (isinstance(self.controlnet, ControlNetModel) or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)):
            self.check_image(image, prompt, prompt_embeds)
        elif (isinstance(self.controlnet, MultiControlNetModel) or is_compiled
              and isinstance(self.controlnet._orig_mod, MultiControlNetModel)):
            if not isinstance(image, list):
                raise TypeError(
                    'For multiple controlnets: `image` must be type `list`')
            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError(
                    'A single batch of multiple conditionings are supported at the moment.'
                )
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    'For multiple controlnets: `image` must have the same length as the number of controlnets.'
                )
            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False
        # Check `controlnet_conditioning_scale`
        if (isinstance(self.controlnet, ControlNetModel) or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError(
                    'For single controlnet: `controlnet_conditioning_scale` must be type `float`.'
                )
        elif (isinstance(self.controlnet, MultiControlNetModel) or is_compiled
              and isinstance(self.controlnet._orig_mod, MultiControlNetModel)):
            if isinstance(controlnet_conditioning_scale, list):
                if any(
                        isinstance(i, list)
                        for i in controlnet_conditioning_scale):
                    raise ValueError(
                        'A single batch of multiple conditionings are supported at the moment.'
                    )
            elif isinstance(
                    controlnet_conditioning_scale,
                    list) and len(controlnet_conditioning_scale) != len(
                        self.controlnet.nets):
                raise ValueError(
                    'For multiple controlnets: When `controlnet_conditioning_scale` '
                    'is specified as `list`, it must have'
                    ' the same length as the number of controlnets')
        else:
            assert False

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]
        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            height = (height // 8) * 8  # round down to nearest multiple of 8
        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            width = (width // 8) * 8  # round down to nearest multiple of 8
        return height, width

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_embeddings_multiples=3,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [''] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.')
        if prompt_embeds is None or negative_prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if do_classifier_free_guidance and negative_prompt_embeds is None:
                    negative_prompt = self.maybe_convert_prompt(
                        negative_prompt, self.tokenizer)

            prompt_embeds1, negative_prompt_embeds1 = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt
                if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
            )
            if prompt_embeds is None:
                prompt_embeds = prompt_embeds1
            if negative_prompt_embeds is None:
                negative_prompt_embeds = negative_prompt_embeds1

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
                                           seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def denoise_latents(self, latents, t, prompt_embeds, control_image,
                        controlnet_conditioning_scale, guess_mode,
                        cross_attention_kwargs, do_classifier_free_guidance,
                        guidance_scale, extra_step_kwargs,
                        views_scheduler_status):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat(
            [latents] * 2) if do_classifier_free_guidance else latents
        self.scheduler.__dict__.update(views_scheduler_status[0])
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t)
        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            controlnet_latent_model_input = latents
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            controlnet_latent_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_latent_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )
        if guess_mode and do_classifier_free_guidance:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [
                torch.cat([torch.zeros_like(d), d])
                for d in down_block_res_samples
            ]
            mid_block_res_sample = torch.cat(
                [torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        return latents

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :,
              y, :] = a[:, :, -blend_extent
                        + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (
                            y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent
                              + x] * (1 - x / blend_extent) + b[:, :, :, x] * (
                                  x / blend_extent)
        return b

    def get_blocks(self, latents, control_image, tile_latent_min_size,
                   overlap_size):
        rows_latents = []
        rows_control_images = []
        for i in range(0, latents.shape[2] - overlap_size, overlap_size):
            row_latents = []
            row_control_images = []
            for j in range(0, latents.shape[3] - overlap_size, overlap_size):
                latents_input = latents[:, :, i:i + tile_latent_min_size,
                                        j:j + tile_latent_min_size]
                c_start_i = self.vae_scale_factor * i
                c_end_i = self.vae_scale_factor * (i + tile_latent_min_size)
                c_start_j = self.vae_scale_factor * j
                c_end_j = self.vae_scale_factor * (j + tile_latent_min_size)
                control_image_input = control_image[:, :, c_start_i:c_end_i,
                                                    c_start_j:c_end_j]
                row_latents.append(latents_input)
                row_control_images.append(control_image_input)
            rows_latents.append(row_latents)
            rows_control_images.append(row_control_images)
        return rows_latents, rows_control_images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image,
                     List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image,
                             List[torch.FloatTensor],
                             List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor],
                                    None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        context_size: int = 768,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/
                src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting
                than for [`~StableDiffusionControlNetPipeline.__call__`].
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            context_size ('int', *optional*, defaults to '768'):
                tiled size when denoise the latents.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        def tiled_decode(
            self,
            z: torch.FloatTensor,
            return_dict: bool = True
        ) -> Union[DecoderOutput, torch.FloatTensor]:
            r"""Decode a batch of images using a tiled decoder.

            Args:
            When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
            steps. This is useful to keep memory use constant regardless of image size. The end result of tiled
            decoding is: different from non-tiled decoding due to each tile using a different decoder.
            To avoid tiling artifacts, the tiles overlap and are blended together to form a smooth output.
            You may still see tile-sized changes in the look of the output, but they should be much less noticeable.
                z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
                `True`):
                    Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            """
            _tile_overlap_factor = 1 - self.tile_overlap_factor
            overlap_size = int(self.tile_latent_min_size
                               * _tile_overlap_factor)
            blend_extent = int(self.tile_sample_min_size
                               * self.tile_overlap_factor)
            row_limit = self.tile_sample_min_size - blend_extent
            w = z.shape[3]
            z = torch.cat([z, z[:, :, :, :w // 4]], dim=-1)
            # Split z into overlapping 64x64 tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.

            rows = []
            for i in range(0, z.shape[2], overlap_size):
                row = []
                tile = z[:, :, i:i + self.tile_latent_min_size, :]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                vae_scale_factor = decoded.shape[-1] // tile.shape[-1]
                row.append(decoded)
                rows.append(row)
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(
                        self.blend_h(
                            tile[:, :, :row_limit, w * vae_scale_factor:],
                            tile[:, :, :row_limit, :w * vae_scale_factor],
                            tile.shape[-1] - w * vae_scale_factor))
                result_rows.append(torch.cat(result_row, dim=3))

            dec = torch.cat(result_rows, dim=2)
            if not return_dict:
                return (dec, )

            return DecoderOutput(sample=dec)

        self.vae.tiled_decode = tiled_decode.__get__(self.vae, AutoencoderKL)

        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        controlnet = self.controlnet._orig_mod if is_compiled_module(
            self.controlnet) else self.controlnet

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
                controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale
                                             ] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions if isinstance(
                controlnet, ControlNetModel) else
            controlnet.nets[0].config.global_pool_conditions)
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # 4. Prepare image, and controlnet_conditioning_image
        image = prepare_image(image)

        # 5. Prepare image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size
                                               * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)]
        # value = torch.zeros_like(latents)
        _, _, height, width = control_image.size()
        tile_latent_min_size = context_size // self.vae_scale_factor
        tile_overlap_factor = 0.5
        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
        blend_extent = int(tile_latent_min_size * tile_overlap_factor)
        row_limit = tile_latent_min_size - blend_extent
        w = latents.shape[3]
        latents = torch.cat([latents, latents[:, :, :, :overlap_size]], dim=-1)
        control_image_extend = control_image[:, :, :, :overlap_size
                                             * self.vae_scale_factor]
        control_image = torch.cat([control_image, control_image_extend],
                                  dim=-1)

        # 8. Denoising loop
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents_input, control_image_input = self.get_blocks(
                    latents, control_image, tile_latent_min_size, overlap_size)
                rows = []
                for latents_input_, control_image_input_ in zip(
                        latents_input, control_image_input):
                    num_block = len(latents_input_)
                    # get batched latents_input
                    latents_input_ = torch.cat(
                        latents_input_[:num_block], dim=0)
                    # get batched prompt_embeds
                    prompt_embeds_ = torch.cat(
                        [prompt_embeds.chunk(2)[0]] * num_block
                        + [prompt_embeds.chunk(2)[1]] * num_block,
                        dim=0)
                    # get batched control_image_input
                    control_image_input_ = torch.cat(
                        [
                            x[0, :, :, ][None, :, :, :]
                            for x in control_image_input_[:num_block]
                        ] + [
                            x[1, :, :, ][None, :, :, :]
                            for x in control_image_input_[:num_block]
                        ],
                        dim=0)
                    latents_output = self.denoise_latents(
                        latents_input_, t, prompt_embeds_,
                        control_image_input_, controlnet_conditioning_scale,
                        guess_mode, cross_attention_kwargs,
                        do_classifier_free_guidance, guidance_scale,
                        extra_step_kwargs, views_scheduler_status)
                    rows.append(list(latents_output.chunk(num_block)))
                result_rows = []
                for i, row in enumerate(rows):
                    result_row = []
                    for j, tile in enumerate(row):
                        # blend the above tile and the left tile
                        # to the current tile and add the current tile to the result row
                        if i > 0:
                            tile = self.blend_v(rows[i - 1][j], tile,
                                                blend_extent)
                        if j > 0:
                            tile = self.blend_h(row[j - 1], tile, blend_extent)
                        if j == 0:
                            tile = self.blend_h(row[-1], tile, blend_extent)
                        if i != len(rows) - 1:
                            if j == len(row) - 1:
                                result_row.append(tile[:, :, :row_limit, :])
                            else:
                                result_row.append(
                                    tile[:, :, :row_limit, :row_limit])
                        else:
                            if j == len(row) - 1:
                                result_row.append(tile[:, :, :, :])
                            else:
                                result_row.append(tile[:, :, :, :row_limit])
                    result_rows.append(torch.cat(result_row, dim=3))
                latents = torch.cat(result_rows, dim=2)

                # call the callback, if provided
                condition_i = i == len(timesteps) - 1
                condition_warm = (i + 1) > num_warmup_steps and (
                    i + 1) % self.scheduler.order == 0
                if condition_i or condition_warm:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
            latents = latents[:, :, :, :w]

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.unet.to('cpu')
            self.controlnet.to('cpu')
            torch.cuda.empty_cache()

        if not output_type == 'latent':
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)

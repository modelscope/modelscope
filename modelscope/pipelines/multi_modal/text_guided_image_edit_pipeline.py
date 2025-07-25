import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Union

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import load_image
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

@PIPELINES.register_module(
    Tasks.text_guided_image_editing, module=Pipelines.text_guided_image_editing)
class TextGuidedImageEditingPipeline(Pipeline):
    """ 文本引导的图像编辑管道
    使用自然语言指令修改图像内容
    
    示例：
        >>> from modelscope.pipelines import pipeline
        >>> editor = pipeline('text-guided-image-editing', 'AI-ModelScope/instruct-pix2pix')
        >>> result = editor(
                image='input.jpg',
                instruction='将汽车替换为自行车',
                style_prompt='赛博朋克风格'  # 可选
            )
        >>> result[OutputKeys.OUTPUT_IMAGE].save('output.jpg')
    """

    def __init__(self, model: str, **kwargs):
        """
        使用`model`和`preprocessor`初始化管道
        Args:
            model: 模型id或本地路径
        """
        super().__init__(model=model, **kwargs)
        self.model_dir = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_pipeline()
        logger.info('文本引导图像编辑管道初始化完成')

    def _load_pipeline(self):
        """ 加载预训练的InstructPix2Pix模型 """
        # 初始化组件
        tokenizer = CLIPTokenizer.from_pretrained(self.model_dir, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.model_dir, subfolder="text_encoder")
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(self.model_dir, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(self.model_dir, subfolder="feature_extractor")
        
        # 创建管道
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_dir,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=None,  # 禁用安全检查器以获得更多创作自由
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        # 优化设置
        if self.device == 'cuda':
            self.pipe.enable_attention_slicing()
            self.pipe.enable_xformers_memory_efficient_attention()

    def preprocess(self, inputs: Input) -> Dict[str, Any]:
        """ 预处理输入图像和文本指令 """
        # 加载图像并转为PIL格式
        image = load_image(inputs['image'])
        
        # 获取文本指令
        instruction = inputs.get('instruction')
        if not instruction:
            raise ValueError('缺少编辑指令文本：`instruction`参数必填')
        
        # 可选风格文本
        style_prompt = inputs.get('style_prompt', '')
        
        # 组合提示词：指令 + 风格（若有）
        text_prompt = f"{instruction}, {style_prompt}" if style_prompt else instruction
        
        # 图像预处理：调整大小和归一化
        processed_image = self.pipe.feature_extractor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        return {
            'original_image': image,
            'processed_image': processed_image,
            'prompt': text_prompt,
            'num_inference_steps': inputs.get('num_inference_steps', 20),
            'image_guidance_scale': inputs.get('image_guidance_scale', 1.5),
            'guidance_scale': inputs.get('guidance_scale', 7.5),
            'seed': inputs.get('seed', None)
        }

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ 执行图像编辑 """
        # 设置随机种子
        generator = None
        if inputs['seed'] is not None:
            generator = torch.Generator(device=self.device).manual_seed(inputs['seed'])
        
        # 执行图像编辑
        edited_image = self.pipe(
            prompt=inputs['prompt'],
            image=inputs['processed_image'],
            num_inference_steps=inputs['num_inference_steps'],
            image_guidance_scale=inputs['image_guidance_scale'],
            guidance_scale=inputs['guidance_scale'],
            generator=generator
        ).images[0]
        
        return {'edited_image': edited_image}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ 后处理：返回编辑后的图像 """
        return {OutputKeys.OUTPUT_IMAGE: inputs['edited_image']}

    def __call__(self, 
                image: Union[str, Image.Image], 
                instruction: str,
                style_prompt: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """ 调用入口
        Args:
            image: 输入图像路径或PIL对象
            instruction: 编辑指令文本（必选）
            style_prompt: 可选风格描述文本
            kwargs: 其他参数（如num_inference_steps, guidance_scale等）
        Returns:
            包含编辑后图像的字典（键：OutputKeys.OUTPUT_IMAGE）
        """
        return super().__call__(
            {'image': image, 'instruction': instruction, 'style_prompt': style_prompt, **kwargs}
        )
import os
import os.path as osp
import random
import torch
from PIL import Image
import torch.cuda.amp as amp
from copy import copy
from typing import Any, Dict

from modelscope.utils.constant import ModelFile, Tasks
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.logger import get_logger

from modelscope.models.multi_modal.image_to_video.modules import *
from modelscope.models.multi_modal.image_to_video.utils.config import cfg
from modelscope.models.multi_modal.image_to_video.utils.diffusion import GaussianDiffusion
import modelscope.models.multi_modal.image_to_video.utils.transforms as data
from modelscope.models.multi_modal.image_to_video.utils.seed import setup_seed
from modelscope.models.multi_modal.image_to_video.utils.shedule import beta_schedule
from modelscope.models.multi_modal.image_to_video.utils.registry_class import UNET, EMBEDDER, AUTO_ENCODER

__all__ = ['ImageToVideo']

logger = get_logger()

@MODELS.register_module(Tasks.image_to_video_task, module_name=Models.image_to_video_model)
class ImageToVideo(TorchModel):
    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir=model_dir, *args, **kwargs)
        
        self.config = Config.from_file(osp.join(model_dir, ModelFile.CONFIGURATION))
        
        # assign default value
        cfg.batch_size = 1
        cfg.target_fps = 8
        cfg.max_frames = 32
        cfg.latent_hei = 32
        cfg.latent_wid = 56
        cfg.model_path = osp.join(model_dir, self.config.model.model_args.ckpt_unet)
    
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if 'seed' in self.config.model.model_args.keys():
            cfg.seed = self.config.model.model_args.seed
        else:
            cfg.seed = random.randint(0, 99999) 
        setup_seed(cfg.seed)
        
        # transform
        vid_trans = data.Compose([
            data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
            data.Resize(cfg.vit_resolution),
            data.ToTensor(),
            data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])
        self.vid_trans = vid_trans
        
        cfg.embedder.pretrained = osp.join(model_dir, self.config.model.model_args.ckpt_clip)
        clip_encoder = EMBEDDER.build(cfg.embedder)
        clip_encoder.model.to(self.device)
        self.clip_encoder = clip_encoder
        logger.info(f'Build encoder with {cfg.embedder.type}')

        # [unet]
        generator = UNET.build(cfg.UNet)
        generator = generator.to(self.device)
        generator.eval()
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        ret = generator.load_state_dict(load_dict['state_dict'], strict=True)
        self.generator = generator
        logger.info('Load model {} path {}, with local status {}'.format(cfg.UNet.type, cfg.model_path, ret))
        
        # [diffusion]
        betas = beta_schedule('linear_sd', cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
        diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg.mean_type,
            var_type=cfg.var_type,
            loss_type=cfg.loss_type,
            rescale_timesteps=False,
            noise_strength=getattr(cfg, 'noise_strength', 0))
        self.diffusion = diffusion
        logger.info('Build diffusion with type of GaussianDiffusion')

        # [auotoencoder]
        cfg.auto_encoder.pretrained = osp.join(model_dir, self.config.model.model_args.ckpt_autoencoder)
        autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
        autoencoder.eval() # freeze
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.to(self.device)
        self.autoencoder = autoencoder
        torch.cuda.empty_cache()

        zero_feature = torch.zeros(1, 1, cfg.UNet.input_dim).to(self.device)
        self.zero_feature = zero_feature
        self.fps_tensor = torch.tensor([cfg.target_fps], dtype=torch.long, device=self.device)
        self.cfg = cfg

    def forward(self, input: Dict[str, Any]):
        img_path = input['img_path']

        cfg = self.cfg 
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        vit_frame = self.vid_trans(image)
        vit_frame = vit_frame.unsqueeze(0)
        vit_frame = vit_frame.to(self.device)
        img_embedding = self.clip_encoder(vit_frame).unsqueeze(1)

        noise = self.build_noise()
        zero_feature = copy(self.zero_feature)
        with torch.no_grad():
            with amp.autocast(enabled=cfg.use_fp16):
                model_kwargs=[
                    {'y': img_embedding, 'fps': self.fps_tensor}, 
                    {'y': zero_feature.repeat(cfg.batch_size, 1, 1), 'fps': self.fps_tensor}]
                gen_video = self.diffusion.ddim_sample_loop(
                    noise=noise,
                    model=self.generator,
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
        
            gen_video = 1. / cfg.scale_factor * gen_video # [1, 4, 32, 32, 56]
            gen_video = rearrange(gen_video, 'b c f h w -> (b f) c h w')
            chunk_size = min(cfg.decoder_bs, gen_video.shape[0])
            gen_video_list = torch.chunk(gen_video, gen_video.shape[0]//chunk_size, dim=0)
            decode_generator = []
            for vd_data in gen_video_list:
                gen_frames = self.autoencoder.decode(vd_data)
                decode_generator.append(gen_frames)
        
        gen_video = torch.cat(decode_generator, dim=0)
        gen_video = rearrange(gen_video, '(b f) c h w -> b c f h w', b = cfg.batch_size)
        
        return gen_video.type(torch.float32).cpu()
    
    def build_noise(self):
        cfg = self.cfg
        noise = torch.randn([1, 4, cfg.max_frames, cfg.latent_hei, cfg.latent_wid]).to(self.device)
        if cfg.noise_strength > 0:
            b, c, f, *_ = noise.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
            noise = noise + cfg.noise_strength * offset_noise
        return noise.contiguous()

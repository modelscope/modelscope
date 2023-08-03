# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from os import path as osp
from typing import Any, Dict

import Image
import open_clip
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from einops import rearrange

from .config import cfg
from utils.config import Config
from modelscope.models.multi_modal.videocomposer.diffusion import (beta_schedule, GaussianDiffusion)
from modelscope.models.multi_modal.videocomposer.utils import setup_seed
from modelscope.models.multi_modal.videocomposer.utils import to_device
import modelscope.models.multi_modal.videocomposer.models as models
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.videocomposer.autoencoder import (
    AutoencoderKL, DiagonalGaussianDistribution)
from modelscope.models.multi_modal.videocomposer.clip import (
    FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder)
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['VideoComposer']


@torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )
    return scale_factor * z


@MODELS.register_module(
    Tasks.text_to_video_synthesis, module_name=Models.videocomposer)
class VideoComposer(Model):
    r"""
    task for video composer.

    Attributes:
        sd_model: denosing model using in this task.
        diffusion: diffusion model for DDIM.
        autoencoder: decode the latent representation into visual space with VQGAN.
        clip_encoder: encode the text into text embedding.
    """

    def __init__(self, model_dir, *args, **kwargs):
        r"""
        Args:
            model_dir (`str` or `os.PathLike`)
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co
                      or modelscope.cn. Valid model ids can be located at the root-level, like `bert-base-uncased`,
                      or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
        """
        super().__init__(model_dir=model_dir, *args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        clip_checkpoint = kwargs.pop('clip_checkpoint',
                                     'open_clip_pytorch_model.bin')
        sd_checkpoint = kwargs.pop('sd_checkpoint', 'v2-1_512-ema-pruned.ckpt')
        _cfg = Config(load=True)
        cfg.update(_cfg.cfg_dict)
        # Copy update input parameter to current task
        for k, v in cfg_update.items():
            cfg[k] = v
        self.cfg = cfg
        if not 'MASTER_ADDR' in os.environ:
            os.environ['MASTER_ADDR']='localhost'
            os.environ['MASTER_PORT']= find_free_port()
        self.cfg.pmi_rank = int(os.getenv('RANK', 0))
        self.cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
        setup_seed(self.cfg.seed)
        self.read_image = kwargs.pop('read_image', False)
        self.read_style = kwargs.pop('read_style', True)
        self.read_sketch = kwargs.pop('read_sketch', False)
        self.save_origin_video = kwargs.pop('save_origin_video', True)
        self.video_compositions = kwargs.pop('video_compositions', [
            'text', 'mask', 'depthmap', 'sketch', 'motion', 'image',
            'local_image', 'single_sketch'
        ])
        self.viz_num = self.cfg.batch_size
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            layer='penultimate',
            pretrained=os.path.join(model_dir, clip_checkpoint))
        self.clip_encoder = self.clip_encoder.to(self.device)
        self.clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(
            layer='penultimate',
            pretrained=os.path.join(model_dir, clip_checkpoint))
        self.clip_encoder_visual.model.to(self.device)
        self.autoencoder = AutoencoderKL(
            ddconfig, 4, ckpt_path=os.path.join(model_dir, sd_checkpoint))
        self.model = UNetSD_temporal(
            cfg=self.cfg,
            in_dim=self.cfg.unet_in_dim,
            concat_dim=self.cfg.unet_concat_dim,
            dim=self.cfg.unet_dim,
            y_dim=self.cfg.unet_y_dim,
            context_dim=self.cfg.unet_context_dim,
            out_dim=self.cfg.unet_out_dim,
            dim_mult=self.cfg.unet_dim_mult,
            num_heads=self.cfg.unet_num_heads,
            head_dim=self.cfg.unet_head_dim,
            num_res_blocks=self.cfg.unet_res_blocks,
            attn_scales=self.cfg.unet_attn_scales,
            dropout=self.cfg.unet_dropout,
            temporal_attention=self.cfg.temporal_attention,
            temporal_attn_times=self.cfg.temporal_attn_times,
            use_checkpoint=self.cfg.use_checkpoint,
            use_fps_condition=self.cfg.use_fps_condition,
            use_sim_mask=self.cfg.use_sim_mask,
            video_compositions=self.cfg.video_compositions,
            misc_dropout=self.cfg.misc_dropout,
            p_all_zero=self.cfg.p_all_zero,
            p_all_keep=self.cfg.p_all_zero,
            zero_y=zero_y,
            black_image_feature=black_image_feature,
        ).to(self.device)

        # Load checkpoint
        resume_step = 1
        if self.cfg.resume and self.cfg.resume_checkpoint:
            if hasattr(self.cfg, "text_to_video_pretrain") and self.cfg.text_to_video_pretrain:
                ss = torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint))
                ss = {key:p for key,p in ss.items() if 'input_blocks.0.0' not in key}
                model.load_state_dict(ss,strict=False)
            else:
                model.load_state_dict(torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint), map_location='cpu'),strict=False)
            if self.cfg.resume_step:
                resume_step = self.cfg.resume_step
            
            logging.info(f'Successfully load step {resume_step} model from {self.cfg.resume_checkpoint}')
            torch.cuda.empty_cache()
        else:
            logging.error(f'The checkpoint file {self.cfg.resume_checkpoint} is wrong')
            raise ValueError(f'The checkpoint file {self.cfg.resume_checkpoint} is wrong ')

        # diffusion
        betas = beta_schedule('linear_sd', self.cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=self.cfg.mean_type,
            var_type=self.cfg.var_type,
            loss_type=self.cfg.loss_type,
            rescale_timesteps=False)
        

    def forward(self, input: Dict[str, Any]):
        # input: ref_frame, cap_txt, video_data, misc_data, feature_framerate, mask, mv_data, style_image
        zero_y = self.clip_encoder('').detach()
        black_image_feature = self.clip_encoder_visual(
            self.clip_encoder_visual.black_image).unsqueeze(1)
        black_image_feature = torch.zeros_like(black_image_feature)

        frame_in = None
        if self.read_image:
            image_key = self.cfg.image_path
            frame = Image.open(open(image_key, mode='rb')).convert('RGB')
            frame_in = misc_transforms([frame])

        frame_sketch = None
        if self.read_sketch:
            sketch_key = self.cfg.sketch_path
            frame_sketch = Image.open(open(sketch_key,
                                           mode='rb')).convert('RGB')
            frame_sketch = misc_transforms([frame_sketch])  #

        frame_style = None
        if self.read_style:
            frame_style = Image.open(open(input['style_image'],
                                          mode='rb')).convert('RGB')

        # Generators for various conditions
        if 'depthmap' in self.video_compositions:
            midas = models.midas_v3(
                pretrained=True).eval().requires_grad_(False).to(
                    memory_format=torch.channels_last).half().to(self.device)
        if 'canny' in self.video_compositions:
            canny_detector = CannyDetector()
        if 'sketch' in self.video_compositions:
            pidinet = pidinet_bsd(
                pretrained=True,
                vanilla_cnn=True).eval().requires_grad_(False).to(self.device)
            cleaner = sketch_simplification_gan(
                pretrained=True).eval().requires_grad_(False).to(self.device)
            pidi_mean = torch.tensor(self.sketch_mean).view(1, -1, 1,
                                                            1).to(self.device)
            pidi_std = torch.tensor(self.sketch_std).view(1, -1, 1,
                                                          1).to(self.device)
        # Placeholder for color inference
        palette = None

        # auotoencoder
        ddconfig = {'double_z': True, 'z_channels': 4, \
                    'resolution': 256, 'in_channels': 3, \
                    'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], \
                    'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        # autoencoder = AutoencoderKL(ddconfig, 4, ckpt_path=DOWNLOAD_TO_CACHE(cfg.sd_checkpoint))
        self.autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.cuda()

        self.model.eval() 
        caps = input['cap_txt']
        if self.cfg.max_frames == 1 and self.cfg.use_image_dataset:
            ref_imgs = input['ref_frame']
            video_data = input['video_data']
            misc_data = input['misc_data']
            mask = input['mask']
            mv_data = input['mv_data']
            fps = torch.tensor([self.cfg.feature_framerate]*self.cfg.batch_size, dtype=torch.long, device=self.device)
        else:
            ref_imgs = input['ref_frame']
            video_data = input['video_data']
            misc_data = input['misc_data']
            mask = input['mask']
            mv_data = input['mv_data']

        # save for visualization
        misc_backups = copy(misc_data)
        misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
        mv_data_video = []
        if 'motion' in self.cfg.video_compositions:
            mv_data_video = rearrange(mv_data, 'b f c h w -> b c f h w')

        # mask images
        masked_video = []
        if 'mask' in self.cfg.video_compositions:
            masked_video = make_masked_images(misc_data.sub(0.5).div_(0.5), mask)
            masked_video = rearrange(masked_video, 'b f c h w -> b c f h w')
        
        # Single Image
        image_local = []
        if 'local_image' in self.cfg.video_compositions:
            frames_num = misc_data.shape[1]
            bs_vd_local = misc_data.shape[0]
            if self.cfg.read_image:
                image_local = frame_in.unsqueeze(0).repeat(bs_vd_local,frames_num,1,1,1).cuda()
            else:
                image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
            image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
        
        # encode the video_data
        bs_vd = video_data.shape[0]
        video_data_origin = video_data.clone() 
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
        misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')

        video_data_list = torch.chunk(video_data, video_data.shape[0]//self.cfg.chunk_size, dim=0)
        misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//self.cfg.chunk_size, dim=0)

        with torch.no_grad():
            decode_data = []
            for vd_data in video_data_list:
                encoder_posterior = autoencoder.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior).detach()
                decode_data.append(tmp)
            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)

            depth_data = []
            if 'depthmap' in self.cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    depth = midas(misc_imgs.sub(0.5).div_(0.5).to(memory_format=torch.channels_last).half())
                    depth = (depth / self.cfg.depth_std).clamp_(0, self.cfg.depth_clamp)
                    depth_data.append(depth)
                depth_data = torch.cat(depth_data, dim = 0)
                depth_data = rearrange(depth_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            canny_data = []
            if 'canny' in self.cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    misc_imgs = rearrange(misc_imgs.clone(), 'k c h w -> k h w c')
                    canny_condition = torch.stack([canny_detector(misc_img) for misc_img in misc_imgs])
                    canny_condition = rearrange(canny_condition, 'k h w c-> k c h w')
                    canny_data.append(canny_condition)
                canny_data = torch.cat(canny_data, dim = 0)
                canny_data = rearrange(canny_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            sketch_data = []
            if 'sketch' in self.cfg.video_compositions:
                sketch_list = misc_data_list
                if self.cfg.read_sketch:
                    sketch_repeat = frame_sketch.repeat(frames_num, 1, 1, 1).cuda()
                    sketch_list = [sketch_repeat]

                for misc_imgs in sketch_list:
                    sketch = pidinet(misc_imgs.sub(pidi_mean).div_(pidi_std))
                    sketch = 1.0 - cleaner(1.0 - sketch)
                    sketch_data.append(sketch)
                sketch_data = torch.cat(sketch_data, dim = 0)
                sketch_data = rearrange(sketch_data, '(b f) c h w -> b c f h w', b = bs_vd)

            single_sketch_data = []
            if 'single_sketch' in self.cfg.video_compositions:
                single_sketch_data = sketch_data.clone()[:, :, :1].repeat(1, 1, frames_num, 1, 1)

        # preprocess for input text descripts
        y = clip_encoder(caps).detach()
        y0 = y.clone()
        
        y_visual = []
        if 'image' in self.cfg.video_compositions:
            with torch.no_grad():
                if self.cfg.read_style:
                    y_visual = clip_encoder_visual(clip_encoder_visual.preprocess(frame_style).unsqueeze(0).cuda()).unsqueeze(0)
                    y_visual0 = y_visual.clone()
                else:
                    ref_imgs = ref_imgs.squeeze(1)
                    y_visual = clip_encoder_visual(ref_imgs).unsqueeze(1)
                    y_visual0 = y_visual.clone()

        with torch.no_grad():
            # Log memory
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Sample images (DDIM)
            with amp.autocast(enabled=self.cfg.use_fp16):
                if self.cfg.share_noise:
                    b, c, f, h, w = video_data.shape
                    noise = torch.randn((viz_num, c, h, w), device=self.device)
                    noise = noise.repeat_interleave(repeats=f, dim=0) 
                    noise = rearrange(noise, '(b f) c h w->b c f h w', b = viz_num) 
                    noise = noise.contiguous()
                else:
                    noise=torch.randn_like(video_data[:viz_num])

                full_model_kwargs=[
                    {'y': y0[:viz_num],
                    "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                    'image': None if len(y_visual) == 0 else y_visual0[:viz_num],
                    'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                    'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                    'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                    'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                    'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                    'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                    'fps': fps[:viz_num]}, 
                    {'y': zero_y.repeat(viz_num,1,1) if not self.cfg.use_fps_condition else torch.zeros_like(y0)[:viz_num],
                    "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                    'image': None if len(y_visual) == 0 else torch.zeros_like(y_visual0[:viz_num]),
                    'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                    'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                    'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                    'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                    'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                    'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                    'fps': fps[:viz_num]}
                ]

                # Save generated videos 
                partial_keys = self.cfg.guidances
                noise_motion = noise.clone()
                model_kwargs = prepare_model_kwargs(partial_keys = partial_keys,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = self.cfg.use_fps_condition)
                video_output = self.diffusion.ddim_sample_loop(
                    noise=noise_motion,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=9.0,
                    ddim_timesteps=self.cfg.ddim_timesteps,
                    eta=0.0)
                
                visualize_with_model_kwargs(model_kwargs = model_kwargs,
                    video_data = video_output,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = self.cfg)

        # return video_data.type(torch.float32).cpu()

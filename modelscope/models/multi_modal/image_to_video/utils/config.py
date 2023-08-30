# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import os.path as osp
from datetime import datetime

import torch
from easydict import EasyDict

cfg = EasyDict(__name__='Config: VideoLDM Decoder')

# ---------------------------work dir--------------------------
cfg.work_dir = 'workspace/'

# ---------------------------Global Variable-----------------------------------
cfg.resolution = [448, 256]
# -----------------------------------------------------------------------------

# ---------------------------Dataset Parameter---------------------------------
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]
cfg.max_words = 1000

# PlaceHolder
cfg.vit_out_dim = 1024
cfg.vit_resolution = [224, 224]
cfg.depth_clamp = 10.0
cfg.misc_size = 384
cfg.depth_std = 20.0

cfg.frame_lens = 32
cfg.sample_fps = 8

cfg.batch_sizes = 1
# -----------------------------------------------------------------------------

# ---------------------------Mode Parameters-----------------------------------
# Diffusion
cfg.schedule = 'cosine'
cfg.num_timesteps = 1000
cfg.mean_type = 'v'
cfg.var_type = 'fixed_small'
cfg.loss_type = 'mse'
cfg.ddim_timesteps = 50
cfg.ddim_eta = 0.0
cfg.clamp = 1.0
cfg.share_noise = False
cfg.use_div_loss = False
cfg.noise_strength = 0.1

# classifier-free guidance
cfg.p_zero = 0.1
cfg.guide_scale = 3.0

# clip vision encoder
cfg.vit_mean = [0.48145466, 0.4578275, 0.40821073]
cfg.vit_std = [0.26862954, 0.26130258, 0.27577711]

# Model
cfg.scale_factor = 0.18215
cfg.use_fp16 = True
cfg.temporal_attention = True
cfg.decoder_bs = 8

cfg.UNet = {
    'type': 'Img2VidSDUNet',
    'in_dim': 4,
    'dim': 320,
    'y_dim': cfg.vit_out_dim,
    'context_dim': 1024,
    'out_dim': 8 if cfg.var_type.startswith('learned') else 4,
    'dim_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'head_dim': 64,
    'num_res_blocks': 2,
    'attn_scales': [1 / 1, 1 / 2, 1 / 4],
    'dropout': 0.1,
    'temporal_attention': cfg.temporal_attention,
    'temporal_attn_times': 1,
    'use_checkpoint': False,
    'use_fps_condition': False,
    'use_sim_mask': False,
    'num_tokens': 4,
    'default_fps': 8,
    'input_dim': 1024
}

cfg.guidances = []

# auotoencoder from stabel diffusion
cfg.auto_encoder = {
    'type': 'AutoencoderKL',
    'ddconfig': {
        'double_z': True,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0
    },
    'embed_dim': 4,
    'pretrained': 'v2-1_512-ema-pruned.ckpt'
}
# clip embedder
cfg.embedder = {
    'type': 'FrozenOpenCLIPVisualEmbedder',
    'layer': 'penultimate',
    'vit_resolution': [224, 224],
    'pretrained': 'open_clip_pytorch_model.bin'
}
# -----------------------------------------------------------------------------

# ---------------------------Training Settings---------------------------------
# training and optimizer
cfg.ema_decay = 0.9999
cfg.num_steps = 600000
cfg.lr = 5e-5
cfg.weight_decay = 0.0
cfg.betas = (0.9, 0.999)
cfg.eps = 1.0e-8
cfg.chunk_size = 16
cfg.alpha = 0.7
cfg.save_ckp_interval = 1000
# -----------------------------------------------------------------------------

# ----------------------------Pretrain Settings---------------------------------
# Default: load 2d pretrain
cfg.fix_weight = False
cfg.load_match = False
cfg.pretrained_checkpoint = 'v2-1_512-ema-pruned.ckpt'
cfg.pretrained_image_keys = 'stable_diffusion_image_key_temporal_attention_x1.json'
cfg.resume_checkpoint = 'img2video_ldm_0779000.pth'
# -----------------------------------------------------------------------------

# -----------------------------Visual-------------------------------------------
# Visual videos
cfg.viz_interval = 1000
cfg.visual_train = {
    'type': 'VisualVideoTextDuringTrain',
}
cfg.visual_inference = {
    'type': 'VisualGeneratedVideos',
}
cfg.inference_list_path = ''

# logging
cfg.log_interval = 100

# Default log_dir
cfg.log_dir = 'workspace/output_data'
# -----------------------------------------------------------------------------

# ---------------------------Others--------------------------------------------
# seed
cfg.seed = 8888
# -----------------------------------------------------------------------------

# This code is borrowed and modified from Human Motion Diffusion Model,
# made publicly available under MIT license at https://github.com/GuyTevet/motion-diffusion-model

from .modules import gaussian_diffusion as gd
from .modules.mdm import MDM
from .modules.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model(cfg):
    model = MDM(
        '',
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep='rot6d',
        glob=True,
        glob_rot=True,
        latent_dim=512,
        ff_size=1024,
        smpl_data_path=cfg.smpl_data_path,
        data_rep='hml_vec',
        dataset='humanml',
        clip_version='ViT-B/32',
        **{
            'cond_mode': 'text',
            'cond_mask_prob': 0.1,
            'action_emb': 'tensor'
        })

    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = cfg.sample_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON
                         if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE
                         if not True else gd.ModelVarType.FIXED_SMALL)
                        if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
    )
    return model, diffusion

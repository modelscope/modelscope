# This code is borrowed and modified from Guided Diffusion Model,
# made publicly available under MIT license
# at https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/disco_project

from modelscope.models.cv.motion_generation.modules.respace import \
    space_timesteps
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion


def create_diffusion(diffusion_config):
    predict_xstart = False
    sigma_small = False
    learn_sigma = True

    steps = diffusion_config['steps']
    timestep_respacing = f'ddim{steps}'
    diffusion_steps = 1000

    rescale_timesteps = True

    betas = gd.get_named_beta_schedule('linear', diffusion_steps)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON
                         if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE
                         if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                        if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps)

    return diffusion

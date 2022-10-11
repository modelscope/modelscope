"""
Part of the implementation is borrowed and modified from LaMa, publicly available at
https://github.com/saic-mdal/lama
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.utils.logger import get_logger
from .modules.adversarial import NonSaturatingWithR1
from .modules.ffc import FFCResNetGenerator
from .modules.perceptual import ResNetPL
from .modules.pix2pixhd import NLayerDiscriminator

LOGGER = get_logger()


class BaseInpaintingTrainingModule(nn.Module):

    def __init__(self,
                 model_dir='',
                 use_ddp=True,
                 predict_only=False,
                 visualize_each_iters=100,
                 average_generator=False,
                 generator_avg_beta=0.999,
                 average_generator_start_step=30000,
                 average_generator_period=10,
                 store_discr_outputs_for_vis=False,
                 **kwargs):
        super().__init__()
        LOGGER.info(
            f'BaseInpaintingTrainingModule init called, predict_only is {predict_only}'
        )

        self.generator = FFCResNetGenerator()
        self.use_ddp = use_ddp

        if not predict_only:
            self.discriminator = NLayerDiscriminator()
            self.adversarial_loss = NonSaturatingWithR1(
                weight=10,
                gp_coef=0.001,
                mask_as_fake_target=True,
                allow_scale_mask=True)

            self.average_generator = average_generator
            self.generator_avg_beta = generator_avg_beta
            self.average_generator_start_step = average_generator_start_step
            self.average_generator_period = average_generator_period
            self.generator_average = None
            self.last_generator_averaging_step = -1
            self.store_discr_outputs_for_vis = store_discr_outputs_for_vis

            self.loss_l1 = nn.L1Loss(reduction='none')

            self.loss_resnet_pl = ResNetPL(weight=30, weights_path=model_dir)

        self.visualize_each_iters = visualize_each_iters
        LOGGER.info('BaseInpaintingTrainingModule init done')

    def forward(self, batch: Dict[str,
                                  torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pass data through generator and obtain at leas 'predicted_image' and 'inpainted' keys"""
        raise NotImplementedError()

    def generator_loss(self,
                       batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(
            self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

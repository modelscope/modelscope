# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import tensorflow as tf

from modelscope.models.base import Model, Tensor
from .loss import content_loss, guided_filter, style_loss, total_variation_loss
from .network import unet_generator


class CartoonModel(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir

    def __call__(
            self,
            input_photo: Dict[str, Tensor],
            input_cartoon: Dict[str, Tensor] = None,
            input_superpixel: Dict[str, Tensor] = None) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input_photo: the preprocessed input photo image
            input_cartoon: the preprocessed input cartoon image
            input_superpixel: the computed input superpixel image

        Returns:
            output_dict: output dict of target ids
        """
        if input_cartoon is None:
            output = unet_generator(input_photo)
            output_cartoon = guided_filter(input_photo, output, r=1)
            return {'output_cartoon': output_cartoon}
        else:
            output = unet_generator(input_photo)
            output_cartoon = guided_filter(input_photo, output, r=1)

            con_loss = content_loss(self.model_dir, input_photo,
                                    output_cartoon, input_superpixel)
            sty_g_loss, sty_d_loss = style_loss(input_cartoon, output_cartoon)
            tv_loss = total_variation_loss(output_cartoon)

            g_loss = 1e-1 * sty_g_loss + 2e2 * con_loss + 1e4 * tv_loss
            d_loss = sty_d_loss

            return {
                'output_cartoon': output_cartoon,
                'g_loss': g_loss,
                'd_loss': d_loss,
            }

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Run the forward pass for a model.

        Args:
            input (Dict[str, Tensor]): the dict of the model inputs for the forward method

        Returns:
            Dict[str, Tensor]: output from the model forward pass
        """

"""
Part of the implementation is borrowed and modified from LaMa, publicly available at
https://github.com/saic-mdal/lama
"""
import bisect

import torch
import torch.nn.functional as F

from modelscope.utils.logger import get_logger
from .base import BaseInpaintingTrainingModule
from .modules.feature_matching import feature_matching_loss, masked_l1_loss

LOGGER = get_logger()


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


class LinearRamp:

    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class LadderRamp:

    def __init__(self, start_iters, values):
        self.start_iters = start_iters
        self.values = values
        assert len(values) == len(start_iters) + 1, (len(values),
                                                     len(start_iters))

    def __call__(self, i):
        segment_i = bisect.bisect_right(self.start_iters, i)
        return self.values[segment_i]


def get_ramp(kind='ladder', **kwargs):
    if kind == 'linear':
        return LinearRamp(**kwargs)
    if kind == 'ladder':
        return LadderRamp(**kwargs)
    raise ValueError(f'Unexpected ramp kind: {kind}')


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):

    def __init__(self,
                 model_dir='',
                 predict_only=False,
                 concat_mask=True,
                 rescale_scheduler_kwargs=None,
                 image_to_discriminator='predicted_image',
                 add_noise_kwargs=None,
                 noise_fill_hole=False,
                 const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None,
                 distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0,
                 fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(model_dir=model_dir, predict_only=predict_only)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(
            **rescale_scheduler_kwargs
        ) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.feature_matching_weight = 100
        self.losses_l1_weight_known = 10
        self.losses_l1_weight_missing = 0
        self.fake_fakes_proba = fake_fakes_proba

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (
            1 - mask) * batch['image']

        batch['mask_for_losses'] = mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.losses_l1_weight_known,
                                  self.losses_l1_weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(
            real_batch=img,
            fake_batch=predicted_img,
            generator=self.generator,
            discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(
            predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(
            real_batch=img,
            fake_batch=predicted_img,
            discr_real_pred=discr_real_pred,
            discr_fake_pred=discr_fake_pred,
            mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.feature_matching_weight > 0:
            need_mask_in_fm = False
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(
                discr_fake_features, discr_real_features,
                mask=mask_for_fm) * self.feature_matching_weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(
            real_batch=batch['image'],
            fake_batch=predicted_img,
            generator=self.generator,
            discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(
            batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(
            predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(
            real_batch=batch['image'],
            fake_batch=predicted_img,
            discr_real_pred=discr_real_pred,
            discr_fake_pred=discr_fake_pred,
            mask=batch['mask'])

        total_loss = (total_loss + adv_discr_loss) * 0.1
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        return total_loss, metrics

    def _do_step(self, batch, optimizer_idx=None):
        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        batch = self(batch)
        total_loss = 0
        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            total_loss, metrics = self.discriminator_loss(batch)

        result = dict(loss=total_loss)
        return result

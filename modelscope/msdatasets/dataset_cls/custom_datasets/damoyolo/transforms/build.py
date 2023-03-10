# Copyright © Alibaba, Inc. and its affiliates.
from modelscope.models.cv.tinynas_detection.damo.augmentations.scale_aware_aug import \
    SA_Aug
from . import transforms as T


def build_transforms(start_epoch,
                     total_epochs,
                     no_aug_epochs,
                     iters_per_epoch,
                     num_workers,
                     batch_size,
                     num_gpus,
                     image_max_range=(640, 640),
                     flip_prob=0.5,
                     image_mean=[0, 0, 0],
                     image_std=[1., 1., 1.],
                     autoaug_dict=None):

    transform = [
        T.Resize(image_max_range),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ]

    if autoaug_dict is not None:
        transform += [
            SA_Aug(iters_per_epoch, start_epoch, total_epochs, no_aug_epochs,
                   batch_size, num_gpus, num_workers, autoaug_dict)
        ]

    transform = T.Compose(transform)

    return transform

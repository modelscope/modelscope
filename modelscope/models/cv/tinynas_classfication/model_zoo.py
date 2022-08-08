# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

from . import master_net


def get_zennet():
    model_plainnet_str = (
        'SuperConvK3BNRELU(3,32,2,1)'
        'SuperResK1K5K1(32,80,2,32,1)SuperResK1K7K1(80,432,2,128,5)'
        'SuperResK1K7K1(432,640,2,192,3)SuperResK1K7K1(640,1008,1,160,5)'
        'SuperResK1K7K1(1008,976,1,160,4)SuperResK1K5K1(976,2304,2,384,5)'
        'SuperResK1K5K1(2304,2496,1,384,5)SuperConvK1BNRELU(2496,3072,1,1)')
    use_SE = False
    num_classes = 1000

    model = master_net.PlainNet(
        num_classes=num_classes,
        plainnet_struct=model_plainnet_str,
        use_se=use_SE)

    return model

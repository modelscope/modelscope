# Copyright (c) Alibaba, Inc. and its affiliates.
class Pix2PixOptions():

    def __init__(self):
        self.gpu_ids = []
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netG = 'resnet_9blocks'
        self.netD = 'basic'
        self.norm = 'instance'
        self.no_dropout = False
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.n_layers_D = 3
        self.gan_mode = 'lsgan'
        self.lr = 0.0002
        self.beta1 = 0.5
        self.isTrain = False
        self.checkpoints_dir = './pix2pix_checkpoints'
        self.name = 'mid_net'
        self.lr_policy = 'linear'
        self.direction = 'AtoB'
        self.lambda_L1 = 100.0
        self.preprocess = 'resize_and_crop'

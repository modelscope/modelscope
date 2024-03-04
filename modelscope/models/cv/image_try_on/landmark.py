# The implementation here is modified based on hrnet,
# originally Apache 2.0 License and publicly available at https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

import logging
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from modelscope.models.cv.body_2d_keypoints.hrnet_basic_modules import (
    BasicBlock, Bottleneck, HighResolutionModule, conv3x3)

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False), nn.BatchNorm2d(in_channels), nn.PReLU())

    def forward(self, x):
        return self.block(x)


class LandmarkNet(nn.Module):

    def __init__(self, cfg, in_channel=3, class_num=3, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(LandmarkNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)

        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        self.active_func = nn.Sigmoid()

        self.downsample = nn.Sequential(
            DownSample(384, 384), DownSample(384, 384),
            nn.AdaptiveAvgPool2d((1, class_num)))

        self.property_conv = nn.Sequential(
            nn.Conv2d(
                384, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(
                192, out_channels=32, kernel_size=1, stride=1, padding=0))

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    layer_config,
                    num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches, block, num_blocks,
                                     num_inchannels, num_channels, fuse_method,
                                     reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)
        property_x = y_list[3]

        x = self.final_layer(y_list[0])
        x = self.active_func(x)

        property_x = self.downsample(property_x)
        property_x = torch.squeeze(self.property_conv(property_x),
                                   2).permute(0, 2, 1)

        return x, property_x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


class VTONLandmark(nn.Module):
    """initialize the try on landmark model
    """

    def __init__(self, **kwargs):
        super(VTONLandmark, self).__init__()
        cfg = {
            'AUTO_RESUME': True,
            'CUDNN': {
                'BENCHMARK': True,
                'DETERMINISTIC': False,
                'ENABLED': True
            },
            'DATA_DIR': '',
            'GPUS': '(0,1,2,3)',
            'OUTPUT_DIR': 'output',
            'LOG_DIR': 'log',
            'WORKERS': 24,
            'PRINT_FREQ': 100,
            'DATASET': {
                'COLOR_RGB': True,
                'DATASET': 'mpii',
                'DATA_FORMAT': 'jpg',
                'FLIP': True,
                'NUM_JOINTS_HALF_BODY': 8,
                'PROB_HALF_BODY': -1.0,
                'ROOT': 'data/mpii/',
                'ROT_FACTOR': 30,
                'SCALE_FACTOR': 0.25,
                'TEST_SET': 'valid',
                'TRAIN_SET': 'train'
            },
            'MODEL': {
                'INIT_WEIGHTS': True,
                'NAME': 'pose_hrnet',
                'NUM_JOINTS': 32,
                'PRETRAINED': 'models/pytorch/imagenet/hrnet_w48-8ef0771d.pth',
                'TARGET_TYPE': 'gaussian',
                'IMAGE_SIZE': [256, 256],
                'HEATMAP_SIZE': [64, 64],
                'SIGMA': 2,
                'EXTRA': {
                    'PRETRAINED_LAYERS': [
                        'conv1', 'bn1', 'conv2', 'bn2', 'layer1',
                        'transition1', 'stage2', 'transition2', 'stage3',
                        'transition3', 'stage4'
                    ],
                    'FINAL_CONV_KERNEL':
                    1,
                    'STAGE2': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4],
                        'NUM_CHANNELS': [48, 96],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 4,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4, 4],
                        'NUM_CHANNELS': [48, 96, 192],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 3,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [4, 4, 4, 4],
                        'NUM_CHANNELS': [48, 96, 192, 384],
                        'FUSE_METHOD': 'SUM'
                    }
                }
            },
            'LOSS': {
                'USE_TARGET_WEIGHT': True
            },
            'TRAIN': {
                'BATCH_SIZE_PER_GPU': 32,
                'SHUFFLE': True,
                'BEGIN_EPOCH': 0,
                'END_EPOCH': 210,
                'OPTIMIZER': 'adam',
                'LR': 0.001,
                'LR_FACTOR': 0.1,
                'LR_STEP': [170, 200],
                'WD': 0.0001,
                'GAMMA1': 0.99,
                'GAMMA2': 0.0,
                'MOMENTUM': 0.9,
                'NESTEROV': False
            },
            'TEST': {
                'BATCH_SIZE_PER_GPU': 32,
                'MODEL_FILE': '',
                'FLIP_TEST': True,
                'POST_PROCESS': True,
                'SHIFT_HEATMAP': True
            },
            'DEBUG': {
                'DEBUG': True,
                'SAVE_BATCH_IMAGES_GT': True,
                'SAVE_BATCH_IMAGES_PRED': True,
                'SAVE_HEATMAPS_GT': True,
                'SAVE_HEATMAPS_PRED': True
            }
        }

        # stem net
        self.stage1Net = LandmarkNet(cfg, in_channel=3, class_num=2)
        self.stage2Net = LandmarkNet(cfg, in_channel=38)

        self.stage = 2

    def forward(self, cloth, person):
        c_landmark, c_property = self.stage1Net(cloth)
        if self.stage == 2:
            pred_class = torch.argmax(c_property, dim=1)
            c_heatmap = F.upsample(
                c_landmark,
                scale_factor=4,
                mode='bilinear',
                align_corners=True)
            c_heatmap = c_heatmap * pred_class.unsqueeze(2).unsqueeze(2)
            input2 = torch.cat([person, cloth, c_heatmap], 1)
            p_landmark, p_property = self.stage2Net(input2)
            return c_landmark, c_property, p_landmark, p_property
        else:
            return c_landmark, c_property

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

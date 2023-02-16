# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


class ResNetEncoder(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 num_layers=18,
                 num_input_images=1,
                 pretrained=False,
                 out_chs=32,
                 stride=8):
        layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block = {
            18: models.resnet.BasicBlock,
            50: models.resnet.Bottleneck
        }[num_layers]
        self.upsample_mode = 'bilinear'
        super(ResNetEncoder, self).__init__(block, layers)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.stride = stride
        if stride == 8:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv1_fusion = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.out_conv = nn.Conv2d(128, out_chs, 3, 1, padding=1)

        elif stride == 4:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv1_fusion = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv2_fusion = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.out_conv = nn.Conv2d(64, out_chs, 3, 1, padding=1)

        else:
            raise NotImplementedError

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # del self.layer3
        del self.layer4
        del self.fc
        del self.avgpool

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            loaded = model_zoo.load_url(
                models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded['conv1.weight'] = torch.cat(
                [loaded['conv1.weight']] * num_input_images,
                1) / num_input_images
            loaded_flilter = {
                k: v
                for k, v in loaded.items()
                if 'layer4' not in k and 'fc' not in k
            }
            try:
                print('load pretrained model from:',
                      models.resnet.model_urls['resnet{}'.format(num_layers)])
                self.load_state_dict(loaded_flilter)
            except Exception as e:
                print(e)
                self.load_state_dict(loaded_flilter, strict=False)

    def forward(self, x):
        feats = {}
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num = len(x)
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feats['s4'] = x
        x = self.layer2(x)
        feats['s8'] = x
        x = self.layer3(x)

        if self.stride == 8:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv1(x)
            x = self.upconv1_fusion(torch.cat([x, feats['s8']], dim=1))
            x = self.out_conv(x)

        elif self.stride == 4:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv1(x)
            x = self.upconv1_fusion(torch.cat([x, feats['s8']], dim=1))

            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv2(x)
            x = self.upconv2_fusion(torch.cat([x, feats['s4']], dim=1))

            x = self.out_conv(x)

        if is_list:
            x = torch.split(x, [batch_dim] * num, dim=0)

        return x


########################################################################################################################
class ResNetEncoderV1(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 num_layers=18,
                 num_input_images=1,
                 pretrained=True,
                 out_chs=32,
                 stride=8):
        layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block = {
            18: models.resnet.BasicBlock,
            50: models.resnet.Bottleneck
        }[num_layers]
        self.upsample_mode = 'nearest'
        super().__init__(block, layers)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.stride = stride
        if stride == 8:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv1_fusion = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.out_conv = nn.Conv2d(128, out_chs, 3, 1, padding=1)

        elif stride == 4:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv1_fusion = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv2_fusion = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.out_conv = nn.Conv2d(64, out_chs, 3, 1, padding=1)

        else:
            raise NotImplementedError

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # del self.layer3
        del self.layer4
        del self.fc
        del self.avgpool

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            loaded = model_zoo.load_url(
                models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded['conv1.weight'] = torch.cat(
                [loaded['conv1.weight']] * num_input_images,
                1) / num_input_images
            loaded_flilter = {
                k: v
                for k, v in loaded.items()
                if 'layer4' not in k and 'fc' not in k
            }
            try:
                print('load pretrained model from:',
                      models.resnet.model_urls['resnet{}'.format(num_layers)])
                self.load_state_dict(loaded_flilter)
            except Exception as e:
                print(e)
                self.load_state_dict(loaded_flilter, strict=False)

    def forward(self, x):
        feats = {}
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num = len(x)
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feats['s4'] = x
        x = self.layer2(x)
        feats['s8'] = x
        x = self.layer3(x)

        if self.stride == 8:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv1(x)
            x = self.upconv1_fusion(torch.cat([x, feats['s8']], dim=1))
            x = self.out_conv(x)

        elif self.stride == 4:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv1(x)
            x = self.upconv1_fusion(torch.cat([x, feats['s8']], dim=1))

            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv2(x)
            x = self.upconv2_fusion(torch.cat([x, feats['s4']], dim=1))

            x = self.out_conv(x)

        if is_list:
            x = torch.split(x, [batch_dim] * num, dim=0)

        return x


########################################################################################################################
class ResNetEncoderV2(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 num_layers=18,
                 num_input_images=1,
                 pretrained=True,
                 out_chs=32,
                 stride=8):
        layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block = {
            18: models.resnet.BasicBlock,
            50: models.resnet.Bottleneck
        }[num_layers]
        self.upsample_mode = 'bilinear'
        super().__init__(block, layers)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # del self.layer4
        self.upconv0 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.upconv0_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, padding=1), nn.ReLU(inplace=True))

        self.stride = stride
        if stride == 8:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv1_fusion = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.out_conv = nn.Conv2d(128, out_chs, 3, 1, padding=1)

        elif stride == 4:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv1_fusion = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.upconv2_fusion = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
            self.out_conv = nn.Conv2d(64, out_chs, 3, 1, padding=1)

        else:
            raise NotImplementedError

        del self.fc
        del self.avgpool

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            loaded = model_zoo.load_url(
                models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded['conv1.weight'] = torch.cat(
                [loaded['conv1.weight']] * num_input_images,
                1) / num_input_images
            loaded_flilter = {
                k: v
                for k, v in loaded.items()
                if 'layer4' not in k and 'fc' not in k
            }
            try:
                print('load pretrained model from:',
                      models.resnet.model_urls['resnet{}'.format(num_layers)])
                self.load_state_dict(loaded_flilter)
            except Exception as e:
                print(e)
                self.load_state_dict(loaded_flilter, strict=False)

    def forward(self, x):
        feats = {}
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num = len(x)
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feats['s4'] = x
        x = self.layer2(x)
        feats['s8'] = x
        x = self.layer3(x)

        feats['s16'] = x
        x = self.layer4(x)
        x = self.upconv0(x)
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        x = self.upconv0_fusion(torch.cat([x, feats['s16']], dim=1))

        if self.stride == 8:
            x = self.upconv1(x)
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv1_fusion(torch.cat([x, feats['s8']], dim=1))
            x = self.out_conv(x)

        elif self.stride == 4:
            x = self.upconv1(x)
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv1_fusion(torch.cat([x, feats['s8']], dim=1))

            x = self.upconv2(x)
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            x = self.upconv2_fusion(torch.cat([x, feats['s4']], dim=1))

            x = self.out_conv(x)

        if is_list:
            x = torch.split(x, [batch_dim] * num, dim=0)

        return x


################################################################################
class FeatBlock(nn.Module):

    def __init__(self, planes=128, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, padding=1)
        self.conv2 = nn.Conv2d(planes, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(self.relu(x)))
        x = self.conv2(x)
        return x

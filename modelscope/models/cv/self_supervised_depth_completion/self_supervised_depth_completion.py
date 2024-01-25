# import argparse
import os
import sys
import time
# import mmcv
from argparse import ArgumentParser
# import torchvision
from os import makedirs

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.self_supervised_depth_completion import (criteria,
                                                                   helper)
from modelscope.models.cv.self_supervised_depth_completion.dataloaders.kitti_loader import (
    KittiDepth, input_options, load_calib, oheight, owidth)
from modelscope.models.cv.self_supervised_depth_completion.inverse_warp import (
    Intrinsics, homography_from)
from modelscope.models.cv.self_supervised_depth_completion.metrics import (
    AverageMeter, Result)
from modelscope.models.cv.self_supervised_depth_completion.model import \
    DepthCompletionNet
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from modelscope.utils.config import Config

m_logger = get_logger()


@MODELS.register_module(
    Tasks.self_supervised_depth_completion,
    module_name=Models.self_supervised_depth_completion)
class SelfSupervisedDepthCompletion(TorchModel):
    """SelfSupervisedDepthCompletion Class"""

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)
        parser = ArgumentParser(description='Testing script parameters')
        self.add_args(parser)

        args = parser.parse_args()

        # define loss functions
        self.depth_criterion = criteria.MaskedMSELoss()
        self.photometric_criterion = criteria.PhotometricLoss()
        self.smoothness_criterion = criteria.SmoothnessLoss()

        # args.use_pose = ('photo' in args.train_mode)
        args.use_pose = True
        # args.pretrained = not args.no_pretrained
        args.use_rgb = ('rgb' in args.input) or args.use_pose
        args.use_d = 'd' in args.input
        args.use_g = 'g' in args.input

        args.evaluate = os.path.join(self.model_dir, 'model_best.pth')

        if args.use_pose:
            args.w1, args.w2 = 0.1, 0.1
        else:
            args.w1, args.w2 = 0, 0

        self.cuda = torch.cuda.is_available() and not args.cpu
        if self.cuda:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("=> using '{}' for computation.".format(self.device))

        args_new = args
        if os.path.isfile(args.evaluate):
            print(
                "=> loading checkpoint '{}' ... ".format(args.evaluate),
                end='')
            self.checkpoint = torch.load(
                args.evaluate, map_location=self.device)
            args = self.checkpoint['args']
            args.val = args_new.val
            print('Completed.')
        else:
            print("No model found at '{}'".format(args.evaluate))
            return

        print('=> creating model and optimizer ... ', end='')
        model = DepthCompletionNet(args).to(self.device)
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(
            model_named_params, lr=args.lr, weight_decay=args.weight_decay)
        print('completed.')
        if self.checkpoint is not None:
            model.load_state_dict(self.checkpoint['model'])
            optimizer.load_state_dict(self.checkpoint['optimizer'])
            print('=> checkpoint state loaded.')

        model = torch.nn.DataParallel(model)

        self.model = model
        self.args = args

    def add_args(self, parser):
        """add args."""
        parser.add_argument(
            '-w',
            '--workers',
            default=4,
            type=int,
            metavar='N',
            help='number of data loading workers (default: 4)')
        parser.add_argument(
            '--epochs',
            default=11,
            type=int,
            metavar='N',
            help='number of total epochs to run (default: 11)')
        parser.add_argument(
            '--start-epoch',
            default=0,
            type=int,
            metavar='N',
            help='manual epoch number (useful on restarts)')
        parser.add_argument(
            '-c',
            '--criterion',
            metavar='LOSS',
            default='l2',
            choices=criteria.loss_names,
            help='loss function: | '.join(criteria.loss_names)
            + ' (default: l2)')
        parser.add_argument(
            '-b',
            '--batch-size',
            default=1,
            type=int,
            help='mini-batch size (default: 1)')
        parser.add_argument(
            '--lr',
            '--learning-rate',
            default=1e-5,
            type=float,
            metavar='LR',
            help='initial learning rate (default 1e-5)')
        parser.add_argument(
            '--weight-decay',
            '--wd',
            default=0,
            type=float,
            metavar='W',
            help='weight decay (default: 0)')
        parser.add_argument(
            '--print-freq',
            '-p',
            default=10,
            type=int,
            metavar='N',
            help='print frequency (default: 10)')
        parser.add_argument(
            '--resume',
            default='',
            type=str,
            metavar='PATH',
            help='path to latest checkpoint (default: none)')
        parser.add_argument(
            '--data-folder',
            default='../data',
            type=str,
            metavar='PATH',
            help='data folder (default: none)')
        parser.add_argument(
            '-i',
            '--input',
            type=str,
            default='gd',
            choices=input_options,
            help='input: | '.join(input_options))
        parser.add_argument(
            '-l',
            '--layers',
            type=int,
            default=34,
            help='use 16 for sparse_conv; use 18 or 34 for resnet')
        parser.add_argument(
            '--pretrained',
            action='store_true',
            help='use ImageNet pre-trained weights')
        parser.add_argument(
            '--val',
            type=str,
            default='select',
            choices=['select', 'full'],
            help='full or select validation set')
        parser.add_argument(
            '--jitter',
            type=float,
            default=0.1,
            help='color jitter for images')
        parser.add_argument(
            '--rank-metric',
            type=str,
            default='rmse',
            choices=[m for m in dir(Result()) if not m.startswith('_')],
            help='metrics for which best result is sbatch_datacted')
        parser.add_argument(
            '-e', '--evaluate', default='', type=str, metavar='PATH')
        parser.add_argument('--cpu', action='store_true', help='run on cpu')

    def iterate(self, mode, args, loader, model, optimizer, logger, epoch):
        """iterate data"""
        block_average_meter = AverageMeter()
        average_meter = AverageMeter()
        meters = [block_average_meter, average_meter]
        merged_img = None
        # switch to appropriate mode
        assert mode in ['train', 'val', 'eval', 'test_prediction', 'test_completion'], \
            'unsupported mode: {}'.format(mode)
        model.eval()
        lr = 0

        for i, batch_data in enumerate(loader):
            start = time.time()
            batch_data = {
                key: val.to(self.device)
                for key, val in batch_data.items() if val is not None
            }
            gt = batch_data[
                'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
            data_time = time.time() - start

            start = time.time()
            pred = model(batch_data)
            photometric_loss = 0
            gpu_time = time.time() - start

            # measure accuracy and record loss
            with torch.no_grad():
                mini_batch_size = next(iter(batch_data.values())).size(0)
                result = Result()
                if mode != 'test_prediction' and mode != 'test_completion':
                    result.evaluate(pred.data, gt.data, photometric_loss)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]
                logger.conditional_print(mode, i, epoch, lr, len(loader),
                                         block_average_meter, average_meter)
                merged_img = logger.conditional_save_img_comparison(
                    mode, i, batch_data, pred, epoch)
                merged_img = cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR)
                logger.conditional_save_pred(mode, i, pred, epoch)

        avg = logger.conditional_save_info(mode, average_meter, epoch)
        is_best = logger.rank_conditional_save_best(mode, avg, epoch)
        logger.save_img_comparison_as_best(mode, epoch)
        logger.conditional_summarize(mode, avg, is_best)

        return avg, is_best, merged_img

    def forward(self, source_dir):
        """main function"""

        args = self.args
        args.data_folder = source_dir
        args.result = os.path.join(args.data_folder, 'results')
        if args.use_pose:
            # hard-coded KITTI camera intrinsics
            K = load_calib(args)
            fu, fv = float(K[0, 0]), float(K[1, 1])
            cu, cv = float(K[0, 2]), float(K[1, 2])
            kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
            if self.cuda:
                kitti_intrinsics = kitti_intrinsics.cuda()

        # Data loading code
        print('=> creating data loaders ... ')
        val_dataset = KittiDepth('val', self.args)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True)  # set batch size to be 1 for validation
        print('\t==> val_loader size:{}'.format(len(val_loader)))

        # create backups and results folder
        logger = helper.logger(self.args)
        if self.checkpoint is not None:
            logger.best_result = self.checkpoint['best_result']

        print('=> starting model evaluation ...')
        result, is_best, merged_img = self.iterate('val', self.args,
                                                   val_loader, self.model,
                                                   None, logger,
                                                   self.checkpoint['epoch'])
        return merged_img

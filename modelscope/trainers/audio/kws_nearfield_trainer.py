# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import datetime
import math
import os
import random
import re
import sys
from shutil import copyfile
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from torch import nn as nn
from torch import optim as optim
from torch.distributed import ReduceOp
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from modelscope.metainfo import Trainers
from modelscope.models import Model, TorchModel
from modelscope.msdatasets.task_datasets.audio.kws_nearfield_dataset import \
    kws_nearfield_dataset
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.audio.audio_utils import update_conf
from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.data_utils import to_device
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import (get_dist_info, get_local_rank,
                                          init_dist, is_master,
                                          set_random_seed)
from .kws_utils.batch_utils import executor_cv, executor_test, executor_train
from .kws_utils.det_utils import compute_det
from .kws_utils.file_utils import query_tokens_id, read_lexicon, read_token
from .kws_utils.model_utils import (average_model, convert_to_kaldi,
                                    count_parameters)

logger = get_logger()


@TRAINERS.register_module(
    module_name=Trainers.speech_kws_fsmn_char_ctc_nearfield)
class KWSNearfieldTrainer(BaseTrainer):

    def __init__(self,
                 model: str,
                 work_dir: str,
                 cfg_file: Optional[str] = None,
                 arg_parse_fn: Optional[Callable] = None,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 **kwargs):
        '''
        Args:
            work_dir (str): main directory for training
            kwargs:
                checkpoint (str): basemodel checkpoint, if None, default to use base.pt in model path
                train_data (int): wave list with kaldi style for training
                cv_data (int): wave list with kaldi style for cross validation
                trans_data (str): transcription list with kaldi style, merge train and cv
                tensorboard_dir (str): path to save tensorboard results,
                                       create 'tensorboard_dir' in work_dir by default
        '''
        if isinstance(model, str):
            self.model_dir = self.get_or_download_model_dir(
                model, model_revision)
            if cfg_file is None:
                cfg_file = os.path.join(self.model_dir,
                                        ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None, 'Config file should not be None if model is not from pretrained!'
            self.model_dir = os.path.dirname(cfg_file)

        super().__init__(cfg_file, arg_parse_fn)
        configs = Config.from_file(cfg_file)

        print(kwargs)
        self.launcher = 'pytorch'
        self.dist_backend = configs.train.get('dist_backend', 'nccl')
        self.tensorboard_dir = kwargs.get('tensorboard_dir', 'tensorboard')
        self.checkpoint = kwargs.get(
            'checkpoint', os.path.join(self.model_dir, 'train/base.pt'))
        self.avg_checkpoint = None

        # 1. get rank info
        set_random_seed(kwargs.get('seed', 666))
        self.get_dist_info()
        logger.info('RANK {}/{}/{}, Master addr:{}, Master port:{}'.format(
            self.world_size, self.rank, self.local_rank, self.master_addr,
            self.master_port))

        self.work_dir = work_dir
        if self.rank == 0:
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)
            logger.info(f'Current working dir is {work_dir}')

        # 2. prepare preset files
        token_file = os.path.join(self.model_dir, 'train/tokens.txt')
        assert os.path.exists(token_file), f'{token_file} is missing'
        self.token_table = read_token(token_file)

        lexicon_file = os.path.join(self.model_dir, 'train/lexicon.txt')
        assert os.path.exists(lexicon_file), f'{lexicon_file} is missing'
        self.lexicon_table = read_lexicon(lexicon_file)

        feature_transform_file = os.path.join(
            self.model_dir, 'train/feature_transform.txt.80dim-l2r2')
        assert os.path.exists(feature_transform_file), \
            f'{feature_transform_file} is missing'
        configs.model['cmvn_file'] = feature_transform_file

        # 3. write config.yaml for inference
        self.configs = configs
        if self.rank == 0:
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)
            saved_config_path = os.path.join(self.work_dir, 'config.yaml')
            with open(saved_config_path, 'w') as fout:
                data = yaml.dump(configs.to_dict())
                fout.write(data)

    def train(self, *args, **kwargs):
        # 1. prepare dataset and dataloader
        assert kwargs['train_data'], 'please config train data in dict kwargs'
        assert kwargs['cv_data'], 'please config cv data in dict kwargs'
        assert kwargs[
            'trans_data'], 'please config transcription data in dict kwargs'
        self.train_data = kwargs['train_data']
        self.cv_data = kwargs['cv_data']
        self.trans_data = kwargs['trans_data']

        train_conf = self.configs['preprocessor']
        cv_conf = copy.deepcopy(train_conf)
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['shuffle'] = False
        self.train_dataset = kws_nearfield_dataset(self.train_data,
                                                   self.trans_data, train_conf,
                                                   self.token_table,
                                                   self.lexicon_table, True)
        self.cv_dataset = kws_nearfield_dataset(self.cv_data, self.trans_data,
                                                cv_conf, self.token_table,
                                                self.lexicon_table, True)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=None,
            pin_memory=kwargs.get('pin_memory', False),
            persistent_workers=True,
            num_workers=self.configs.train.dataloader.workers_per_gpu,
            prefetch_factor=self.configs.train.dataloader.get('prefetch', 2))
        self.cv_dataloader = DataLoader(
            self.cv_dataset,
            batch_size=None,
            pin_memory=kwargs.get('pin_memory', False),
            persistent_workers=True,
            num_workers=self.configs.evaluation.dataloader.workers_per_gpu,
            prefetch_factor=self.configs.evaluation.dataloader.get(
                'prefetch', 2))

        # 2. Init kws model from configs
        self.model = self.build_model(self.configs)
        num_params = count_parameters(self.model)
        if self.rank == 0:
            # print(model)
            logger.warning('the number of model params: {}'.format(num_params))

        # 3. if specify checkpoint, load infos and params
        if self.checkpoint is not None and os.path.exists(self.checkpoint):
            load_checkpoint(self.checkpoint, self.model)
            info_path = re.sub('.pt$', '.yaml', self.checkpoint)
            infos = {}
            if os.path.exists(info_path):
                with open(info_path, 'r') as fin:
                    infos = yaml.load(fin, Loader=yaml.FullLoader)
        else:
            logger.warning('Training with random initialized params')
            infos = {}
        self.start_epoch = infos.get('epoch', -1) + 1
        self.configs['train']['start_epoch'] = self.start_epoch

        lr_last_epoch = infos.get('lr',
                                  self.configs['train']['optimizer']['lr'])
        self.configs['train']['optimizer']['lr'] = lr_last_epoch

        # 4. model placement
        self.device_name = kwargs.get('device', 'gpu')
        if self.world_size > 1:
            self.device_name = f'cuda:{self.local_rank}'
        self.device = create_device(self.device_name)

        if self.world_size > 1:
            assert (torch.cuda.is_available())
            # cuda model is required for nn.parallel.DistributedDataParallel
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        else:
            self.model = self.model.to(self.device)

        # 5. update training config file
        if self.rank == 0:
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)
            saved_config_path = os.path.join(self.work_dir, 'config.yaml')
            with open(saved_config_path, 'w') as fout:
                data = yaml.dump(self.configs.to_dict())
                fout.write(data)

        logger.info('Start training...')

        writer = None
        if self.rank == 0:
            os.makedirs(self.work_dir, exist_ok=True)
            writer = SummaryWriter(
                os.path.join(self.work_dir, self.tensorboard_dir))

        log_interval = self.configs['train'].get('log_interval', 10)

        optim_conf = self.configs['train']['optimizer']
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=optim_conf['lr'],
            weight_decay=optim_conf['weight_decay'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            threshold=0.01,
        )

        final_epoch = None
        if self.start_epoch == 0 and self.rank == 0:
            save_model_path = os.path.join(self.work_dir, 'init.pt')
            save_checkpoint(self.model, save_model_path, None, None, None,
                            False)

        # Start training loop
        logger.info('Start training...')
        training_config = {}
        training_config['grad_clip'] = optim_conf['grad_clip']
        training_config['log_interval'] = log_interval
        training_config['world_size'] = self.world_size
        training_config['rank'] = self.rank
        training_config['local_rank'] = self.local_rank

        max_epoch = self.configs['train']['max_epochs']
        totaltime = datetime.datetime.now()
        for epoch in range(self.start_epoch, max_epoch):
            self.train_dataset.set_epoch(epoch)
            training_config['epoch'] = epoch

            lr = optimizer.param_groups[0]['lr']
            logger.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            executor_train(self.model, optimizer, self.train_dataloader,
                           self.device, writer, training_config)
            cv_loss = executor_cv(self.model, self.cv_dataloader, self.device,
                                  training_config)
            logger.info('Epoch {} EVAL info cv_loss {:.6f}'.format(
                epoch, cv_loss))

            if self.rank == 0:
                save_model_path = os.path.join(self.work_dir,
                                               '{}.pt'.format(epoch))
                save_checkpoint(self.model, save_model_path, None, None, None,
                                False)

                info_path = re.sub('.pt$', '.yaml', save_model_path)
                info_dict = dict(
                    epoch=epoch,
                    lr=lr,
                    cv_loss=cv_loss,
                )
                with open(info_path, 'w') as fout:
                    data = yaml.dump(info_dict)
                    fout.write(data)

                writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
                writer.add_scalar('epoch/lr', lr, epoch)
            final_epoch = epoch
            lr_scheduler.step(cv_loss)

        if final_epoch is not None and self.rank == 0:
            writer.close()

        totaltime = datetime.datetime.now() - totaltime
        logger.info('Total time spent: {:.2f} hours'.format(
            totaltime.total_seconds() / 3600.0))

    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        '''
        Args:
            checkpoint_path (str): evaluating with ckpt or default average ckpt
            kwargs:
                test_dir (str): local path for saving test results
                test_data (str): wave list with kaldi style
                trans_data (str): transcription list with kaldi style
                average_num (int): the NO. to do model averaging(checkpoint_path==None)
                batch_size (int): batch size during evaluating
                keywords (str): keyword string, split with ','
                gpu (int): evaluating with cpu/gpu: -1 for cpu; >=0 for gpu,
                           os.environ['CUDA_VISIBLE_DEVICES'] will be setted
        '''
        # 1. get checkpoint
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.warning(
                f'evaluating with specific model: {checkpoint_path}')
            eval_checkpoint = checkpoint_path
        else:
            if self.avg_checkpoint is None:
                avg_num = kwargs.get('average_num', 5)
                self.avg_checkpoint = os.path.join(self.work_dir,
                                                   f'avg_{avg_num}.pt')
                logger.warning(
                    f'default average model not exist: {self.avg_checkpoint}')
                avg_kwargs = dict(
                    dst_model=self.avg_checkpoint,
                    src_path=self.work_dir,
                    val_best=True,
                    avg_num=avg_num,
                )
                self.avg_checkpoint = average_model(**avg_kwargs)

                model_cvt = self.build_model(self.configs)
                kaldi_cvt = convert_to_kaldi(
                    model_cvt,
                    self.avg_checkpoint,
                    self.work_dir,
                )
                logger.warning(
                    f'average model convert to kaldi network: {kaldi_cvt}')

            eval_checkpoint = self.avg_checkpoint
            logger.warning(
                f'evaluating with average model: {self.avg_checkpoint}')

        # 2. get test data and trans
        if kwargs.get('test_data', None) is not None and \
           kwargs.get('trans_data', None) is not None:
            logger.warning('evaluating with specific data and transcription')
            test_data = kwargs['test_data']
            trans_data = kwargs['trans_data']
        else:
            logger.warning(
                'evaluating with cross validation data during training')
            test_data = self.cv_data
            trans_data = self.trans_data
        logger.warning(f'test data: {test_data}')
        logger.warning(f'trans data: {trans_data}')

        # 3. prepare dataset and dataloader
        test_conf = copy.deepcopy(self.configs['preprocessor'])
        test_conf['filter_conf']['max_length'] = 102400
        test_conf['filter_conf']['min_length'] = 0
        test_conf['speed_perturb'] = False
        test_conf['spec_aug'] = False
        test_conf['shuffle'] = False
        test_conf['feature_extraction_conf']['dither'] = 0.0
        if kwargs.get('batch_size', None) is not None:
            test_conf['batch_conf']['batch_size'] = kwargs['batch_size']

        test_dataset = kws_nearfield_dataset(test_data, trans_data, test_conf,
                                             self.token_table,
                                             self.lexicon_table, False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=None,
            pin_memory=kwargs.get('pin_memory', False),
            persistent_workers=True,
            num_workers=self.configs.evaluation.dataloader.workers_per_gpu,
            prefetch_factor=self.configs.evaluation.dataloader.get(
                'prefetch', 2))

        # 4. parse keywords tokens
        assert kwargs.get('keywords',
                          None) is not None, 'at least one keyword is needed'
        keywords_str = kwargs['keywords']
        keywords_list = keywords_str.strip().replace(' ', '').split(',')
        keywords_token = {}
        keywords_tokenset = {0}
        for keyword in keywords_list:
            ids = query_tokens_id(keyword, self.token_table,
                                  self.lexicon_table)
            keywords_token[keyword] = {}
            keywords_token[keyword]['token_id'] = ids
            keywords_token[keyword]['token_str'] = ''.join('%s ' % str(i)
                                                           for i in ids)
            [keywords_tokenset.add(i) for i in ids]
        logger.warning(f'Token set is: {keywords_tokenset}')

        # 5. build model and load checkpoint
        # support assign specific gpu device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs.get('gpu', -1))
        use_cuda = kwargs.get('gpu', -1) >= 0 and torch.cuda.is_available()

        if kwargs.get('jit_model', None):
            model = torch.jit.load(eval_checkpoint)
            # For script model, only cpu is supported.
            device = torch.device('cpu')
        else:
            # Init kws model from configs
            model = self.build_model(self.configs)
            load_checkpoint(eval_checkpoint, model)
            device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)
        model.eval()

        testing_config = {}
        if kwargs.get('test_dir', None) is not None:
            testing_config['test_dir'] = kwargs['test_dir']
        else:
            base_name = os.path.basename(eval_checkpoint)
            testing_config['test_dir'] = os.path.join(self.work_dir,
                                                      'test_' + base_name)
        self.test_dir = testing_config['test_dir']
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # 6. executing evaluation and get score file
        logger.info('Start evaluating...')
        totaltime = datetime.datetime.now()
        score_file = executor_test(model, test_dataloader, device,
                                   keywords_token, keywords_tokenset,
                                   testing_config)
        totaltime = datetime.datetime.now() - totaltime
        logger.info('Total time spent: {:.2f} hours'.format(
            totaltime.total_seconds() / 3600.0))

        # 7. compute det statistic file with score file
        det_kwargs = dict(
            keywords=keywords_str,
            test_data=test_data,
            trans_data=trans_data,
            score_file=score_file,
        )
        det_results = compute_det(**det_kwargs)
        print(det_results)

    def build_model(self, configs) -> nn.Module:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        override this method in a subclass.

        """
        model = Model.from_pretrained(
            self.model_dir, cfg_dict=configs, training=True)
        if isinstance(model, TorchModel) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def get_dist_info(self):
        if os.getenv('RANK', None) is None:
            os.environ['RANK'] = '0'
        if os.getenv('LOCAL_RANK', None) is None:
            os.environ['LOCAL_RANK'] = '0'
        if os.getenv('WORLD_SIZE', None) is None:
            os.environ['WORLD_SIZE'] = '1'
        if os.getenv('MASTER_ADDR', None) is None:
            os.environ['MASTER_ADDR'] = 'localhost'
        if os.getenv('MASTER_PORT', None) is None:
            os.environ['MASTER_PORT'] = '29500'

        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.master_addr = os.environ['MASTER_ADDR']
        self.master_port = os.environ['MASTER_PORT']

        init_dist(self.launcher, self.dist_backend)
        self.rank, self.world_size = get_dist_info()
        self.local_rank = get_local_rank()

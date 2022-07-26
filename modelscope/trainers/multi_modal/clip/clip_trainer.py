import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from modelscope.metainfo import Trainers
from modelscope.models.base import Model
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger
from .clip_trainer_utils import ImageWithCaptionDataset, get_optimizer

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.clip_multi_modal_embedding)
class CLIPTrainer(BaseTrainer):

    def __init__(self, cfg_file: str, model: str, device_id: int, *args,
                 **kwargs):
        super().__init__(cfg_file)

        self.cfg = Config.from_file(cfg_file)
        self.model = Model.from_pretrained(model)
        self.device_id = device_id
        self.total_epoch = self.cfg.train.epoch
        self.train_batch_size = self.cfg.train.batch_size
        self.val_batch_size = self.cfg.evaluation.batch_size
        self.ckpt_dir = self.cfg.train.ckpt_dir

        self.train_dataset = ImageWithCaptionDataset(
            json_file='{}/{}'.format(self.cfg.dataset.root_dir,
                                     self.cfg.dataset.train_set),
            img_dir=self.cfg.dataset.root_dir,
            phase=ModeKeys.TRAIN)
        self.val_dataset = ImageWithCaptionDataset(
            json_file='{}/{}'.format(self.cfg.dataset.root_dir,
                                     self.cfg.dataset.val_set),
            img_dir=self.cfg.dataset.root_dir,
            phase=ModeKeys.EVAL)

    def train(self, *args, **kwargs):
        assert dist.is_initialized()

        self.model.clip_model.train()
        self.model.clip_model.to(self.device_id)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            self.model.clip_model, device_ids=[
                self.device_id,
            ])

        optimizer = get_optimizer(ddp_model)

        for epoch in range(self.total_epoch):
            train_sampler = DistributedSampler(
                dataset=self.train_dataset, shuffle=True)
            train_sampler.set_epoch(epoch)

            train_params = {
                'pin_memory': True,
                'collate_fn': None,
                'batch_size': self.train_batch_size,
                'shuffle': False,
                'drop_last': True,
                'sampler': train_sampler,
                'num_workers': 8
            }

            train_loader = DataLoader(self.train_dataset, **train_params)

            for batch_idx, (img_tensor, text_str_list,
                            img_id_list) in enumerate(train_loader):
                text_info_list = [
                    self.model.tokenize_text(tmp) for tmp in text_str_list
                ]
                text_ids_tensor = torch.cat([tmp[0] for tmp in text_info_list],
                                            dim=0)
                text_masks_tensor = torch.cat(
                    [tmp[1] for tmp in text_info_list], dim=0)

                img_tensor = img_tensor.to(self.device_id, non_blocking=True)
                img_id_list = img_id_list.to(self.device_id, non_blocking=True)
                text_ids_tensor = text_ids_tensor.to(
                    self.device_id, non_blocking=True)
                text_masks_tensor = text_masks_tensor.to(
                    self.device_id, non_blocking=True)

                loss = ddp_model((img_tensor, text_ids_tensor,
                                  text_masks_tensor, img_id_list),
                                 ModeKeys.TRAIN)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    logger.info(
                        'epoch: {}, train batch {}/{}, loss={:.5f}, logit_scale={:.5f}'
                        .format(epoch, batch_idx, len(train_loader),
                                loss.item(),
                                ddp_model.module.logit_scale.exp().item()))
            if dist.get_rank() == 0:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                torch.save(ddp_model.module.state_dict(),
                           '{}/epoch{}.pth'.format(self.ckpt_dir, epoch))

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        if checkpoint_path is not None:
            checkpoint_params = torch.load(checkpoint_path, 'cpu')
            self.model.clip_model.load_state_dict(checkpoint_params)
        self.model.clip_model.eval()
        self.model.clip_model.to(self.device_id)

        val_params = {
            'collate_fn': None,
            'batch_size': self.val_batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 8
        }
        val_loader = DataLoader(self.val_dataset, **val_params)

        tp_cnt_per_batch = []
        processed_cnt = 0
        with torch.no_grad():
            for batch_idx, (img_tensor, text_str_list,
                            img_id_list) in enumerate(val_loader):
                text_info_list = [
                    self.model.tokenize_text(tmp) for tmp in text_str_list
                ]
                text_ids_tensor = torch.cat([tmp[0] for tmp in text_info_list],
                                            dim=0)
                text_masks_tensor = torch.cat(
                    [tmp[1] for tmp in text_info_list], dim=0)

                img_tensor = img_tensor.to(self.device_id, non_blocking=True)
                img_id_list = img_id_list.to(self.device_id, non_blocking=True)
                text_ids_tensor = text_ids_tensor.to(
                    self.device_id, non_blocking=True)
                text_masks_tensor = text_masks_tensor.to(
                    self.device_id, non_blocking=True)

                img_feat = self.model.clip_model(img_tensor, input_type='img')
                text_feat = self.model.clip_model(
                    (text_ids_tensor, text_masks_tensor), input_type='text')

                sim_mat = text_feat @ img_feat.t()
                text_cnt, img_cnt = sim_mat.shape
                top1_scores, match_ids = torch.max(sim_mat, dim=1)

                match_ids = match_ids.int()
                gt_ids = torch.tensor(range(0, text_cnt)).to(
                    self.device_id, non_blocking=True).int()
                error_cnt = torch.nonzero(match_ids - gt_ids)
                processed_cnt += text_cnt

                tp_cnt_per_batch.append(text_cnt - 1.0 * error_cnt.numel())
                logger.info('current acc: {:.3f}'.format(
                    sum(tp_cnt_per_batch) / processed_cnt))

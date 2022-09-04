import os
from typing import Dict, Optional

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import Model
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModeKeys, ModelFile
from .ofa_trainer_utils import (AdjustLabelSmoothedCrossEntropyCriterion,
                                OFADataset, get_schedule)


@TRAINERS.register_module(module_name=Trainers.ofa_tasks)
class OFATrainer(EpochBasedTrainer):

    def __init__(self, model: str, *args, **kwargs):
        # import pdb
        # pdb.set_trace()
        model = Model.from_pretrained(model)
        model_dir = model.model_dir
        cfg_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
        cfg = Config.from_file(cfg_file)
        dataset = load_dataset(
            cfg.dataset.script,
            data_files=cfg.dataset.hf_dataset,
            sep=cfg.dataset.sep,
        )
        ms_dadaset = MsDataset.from_hf_dataset(dataset)
        # train_dataset = OFADataset(
        #     file_path=cfg.dataset.train_set,
        #     selected_id_keys=cfg.dataset.selected_id_keys,
        #     preprocessor=OfaPreprocessor(
        #         model_dir=model_dir, mode=ModeKeys.TRAIN),
        # )
        # val_dataset = OFADataset(
        #     file_path=cfg.dataset.valid_set,
        #     selected_id_keys=cfg.dataset.selected_id_keys,
        #     preprocessor=OfaPreprocessor(
        #         model_dir=model_dir, mode=ModeKeys.EVAL),
        # )
        epoch_steps = len(ms_dadaset['train']) // (
            cfg.train.gradient_accumulation_steps
            * cfg.train.dataloader.batch_size_per_gpu)
        cfg.train.lr_scheduler.num_train_steps = epoch_steps * cfg.train.max_epochs
        cfg.train.criterion.tokenizer = model.tokenizer
        self.criterion = AdjustLabelSmoothedCrossEntropyCriterion(
            cfg.train.criterion)
        optimizer = build_optimizer(model, cfg=cfg.train.optimizer)
        scheduler_class, scheduler_args = get_schedule(cfg.train.lr_scheduler)
        if scheduler_class is not None:
            lr_scheduler = scheduler_class(**{'optimizer': optimizer},
                                           **scheduler_args)
        else:
            lr_scheduler = None
        super().__init__(
            cfg_file=cfg_file,
            model=model,
            data_collator=collate_fn,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            optimizers=(optimizer, lr_scheduler),
            work_dir=cfg.train.work_dir,
            *args,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        pass

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        pass

    def prediction_step(self, model, inputs):
        pass

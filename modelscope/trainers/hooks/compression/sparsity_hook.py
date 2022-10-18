# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope import __version__
from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.torch_utils import is_master


@HOOKS.register_module(module_name=Hooks.SparsityHook)
class SparsityHook(Hook):

    PRIORITY = Priority.HIGHEST

    def __init__(self, pruning_method, config={}, save_dir=None):
        self.pruning_method = pruning_method
        self.save_dir = save_dir

        self.compress_module = config.get('compress_module', [])
        self.weight_rank = config.get('weight_rank', 8)
        self.weight_beta = config.get('weight_beta', 1)
        self.mask_rank = config.get('mask_rank', 8)
        self.mask_alpha1 = config.get('mask_alpha1', 1)
        self.mask_alpha2 = config.get('mask_alpha2', 1)

        self.step = 0
        self.total_step = 0
        self.frequency = config.get('frequency', 1)
        self.initial_warmup = config.get('initial_warmup', 0.1)
        self.final_warmup = config.get('final_warmup', 0.3)
        self.initial_sparsity = config.get('initial_sparsity', 0.0)
        self.final_sparsity = config.get('final_sparsity', 0.0)

    def before_run(self, trainer):
        import torch

        from .utils import SparseLinear, convert_sparse_network

        if self.save_dir is None:
            self.save_dir = trainer.work_dir

        if len(self.compress_module) == 0:
            convert_sparse_network(
                trainer.model,
                pruning_method=self.pruning_method,
                weight_rank=self.weight_rank,
                weight_beta=self.weight_beta,
                mask_rank=self.mask_rank,
                mask_alpha1=self.mask_alpha1,
                mask_alpha2=self.mask_alpha2,
                logger=trainer.logger,
            )
        else:
            for cm in self.compress_module:
                for name, module in trainer.model.named_modules():
                    if name != cm:
                        continue
                    convert_sparse_network(
                        module,
                        pruning_method=self.pruning_method,
                        weight_rank=self.weight_rank,
                        weight_beta=self.weight_beta,
                        mask_rank=self.mask_rank,
                        mask_alpha1=self.mask_alpha1,
                        mask_alpha2=self.mask_alpha2,
                        logger=trainer.logger,
                    )

        for i in range(len(trainer.optimizer.param_groups)):
            new_train_params = []
            for param in trainer.optimizer.param_groups[i]['params']:
                is_find = False
                for name, module in trainer.model.named_modules():
                    if isinstance(module, SparseLinear):
                        if torch.equal(param.half(),
                                       module.weight.data.half()):
                            is_find = True
                            break

                if not is_find:
                    new_train_params.append(param)

            trainer.optimizer.param_groups[i]['params'] = new_train_params

        new_params = []
        for name, module in trainer.model.named_modules():
            if isinstance(module, SparseLinear):
                new_params.extend(
                    [p for p in module.parameters() if p.requires_grad])

        trainer.optimizer.add_param_group({'params': new_params})

        self.total_step = trainer.iters_per_epoch * trainer._max_epochs

    def before_train_iter(self, trainer):
        from .utils import schedule_sparsity_ratio, update_network_sparsity

        cur_sparsity = schedule_sparsity_ratio(
            self.step,
            self.total_step,
            self.frequency,
            self.initial_warmup,
            self.final_warmup,
            self.initial_sparsity,
            self.final_sparsity,
        )

        update_network_sparsity(trainer.model, cur_sparsity)

        if is_master():
            trainer.logger.info(
                f'Step[{self.step}/{self.total_step}] current sparsity ratio = {cur_sparsity}'
            )

        self.step += 1

    def after_run(self, trainer):
        from .utils import generate_sparse_model

        generate_sparse_model(trainer.model, logger=trainer.logger)

        self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer):
        if is_master():
            trainer.logger.info('Saving checkpoint at final compress')
        cur_save_name = os.path.join(self.save_dir, 'compress_model.pth')
        save_checkpoint(trainer.model, cur_save_name, trainer.optimizer)

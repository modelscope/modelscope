# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Trainers
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import TrainerStages
from modelscope.utils.data_utils import to_device
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.speech_frcrn_ans_cirm_16k)
class ANSTrainer(EpochBasedTrainer):
    """
    A trainer is used for acoustic noise suppression.
    Override train_loop() to use dataset just one time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_loop(self, data_loader):
        """
        Update epoch by step number, based on super method.
        """
        self.invoke_hook(TrainerStages.before_run)
        self._epoch = 0
        kwargs = {}
        self.model.train()
        enumerated = enumerate(data_loader)
        for _ in range(self._epoch, self._max_epochs):
            self.invoke_hook(TrainerStages.before_train_epoch)
            self._inner_iter = 0
            for i, data_batch in enumerated:
                data_batch = to_device(data_batch, self.device)
                self.data_batch = data_batch
                self._inner_iter += 1
                self.invoke_hook(TrainerStages.before_train_iter)
                self.train_step(self.model, data_batch, **kwargs)
                self.invoke_hook(TrainerStages.after_train_iter)
                del self.data_batch
                self._iter += 1
                if self._inner_iter >= self.iters_per_epoch:
                    break

            self.invoke_hook(TrainerStages.after_train_epoch)
            self._epoch += 1

        self.invoke_hook(TrainerStages.after_run)

    def prediction_step(self, model, inputs):
        pass

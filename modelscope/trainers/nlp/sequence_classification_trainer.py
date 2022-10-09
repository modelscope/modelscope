# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from typing import Dict, Optional, Tuple, Union

import numpy as np

from modelscope.metainfo import Trainers
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.logger import get_logger

PATH = None
logger = get_logger(PATH)


@TRAINERS.register_module(module_name=Trainers.bert_sentiment_analysis)
class SequenceClassificationTrainer(BaseTrainer):

    def __init__(self, cfg_file: str, *args, **kwargs):
        """ A trainer is used for Sequence Classification

        Based on Config file (*.yaml or *.json), the trainer trains or evaluates on a dataset

        Args:
            cfg_file (str): the path of config file
        Raises:
            ValueError: _description_
        """
        super().__init__(cfg_file)

    def train(self, *args, **kwargs):
        logger.info('Train')
        ...

    def __attr_is_exist(self, attr: str) -> Tuple[Union[str, bool]]:
        """get attribute from config, if the attribute does exist, return false

        Example:
        >>> self.__attr_is_exist("model path")
        out: (model-path, "/workspace/bert-base-sst2")
        >>> self.__attr_is_exist("model weights")
        out: (model-weights, False)

        Args:
            attr (str): attribute str, "model path" -> config["model"][path]

        Returns:
            Tuple[Union[str, bool]]:[target attribute name, the target attribute or False]
        """
        paths = attr.split(' ')
        attr_str: str = '-'.join(paths)
        target = self.cfg[paths[0]] if hasattr(self.cfg, paths[0]) else None

        for path_ in paths[1:]:
            if not hasattr(target, path_):
                return attr_str, False
            target = target[path_]

        if target and target != '':
            return attr_str, target
        return attr_str, False

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        """evaluate a dataset

        evaluate a dataset via a specific model from the `checkpoint_path` path, if the `checkpoint_path`
        does not exist, read from the config file.

        Args:
            checkpoint_path (Optional[str], optional): the model path. Defaults to None.

        Returns:
            Dict[str, float]: the results about the evaluation
            Example:
            {"accuracy": 0.5091743119266054, "f1": 0.673780487804878}
        """
        import torch
        from easynlp.appzoo import load_dataset
        from easynlp.appzoo.dataset import GeneralDataset
        from easynlp.appzoo.sequence_classification.model import \
            SequenceClassification
        from easynlp.utils import losses
        from sklearn.metrics import f1_score
        from torch.utils.data import DataLoader

        raise_str = 'Attribute {} is not given in config file!'

        metrics = self.__attr_is_exist('evaluation metrics')
        eval_batch_size = self.__attr_is_exist('evaluation batch_size')
        test_dataset_path = self.__attr_is_exist('dataset valid file')

        attrs = [metrics, eval_batch_size, test_dataset_path]
        for attr_ in attrs:
            if not attr_[-1]:
                raise AttributeError(raise_str.format(attr_[0]))

        if not checkpoint_path:
            checkpoint_path = self.__attr_is_exist('evaluation model_path')[-1]
            if not checkpoint_path:
                raise ValueError(
                    'Argument checkout_path must be passed if the evaluation-model_path is not given in config file!'
                )

        max_sequence_length = kwargs.get(
            'max_sequence_length',
            self.__attr_is_exist('evaluation max_sequence_length')[-1])
        if not max_sequence_length:
            raise ValueError(
                'Argument max_sequence_length must be passed '
                'if the evaluation-max_sequence_length does not exist in config file!'
            )

        # get the raw online dataset
        raw_dataset = load_dataset(*test_dataset_path[-1].split('/'))
        valid_dataset = raw_dataset['validation']

        # generate a standard dataloader
        pre_dataset = GeneralDataset(valid_dataset, checkpoint_path,
                                     max_sequence_length)
        valid_dataloader = DataLoader(
            pre_dataset,
            batch_size=eval_batch_size[-1],
            shuffle=False,
            collate_fn=pre_dataset.batch_fn)

        # generate a model
        model = SequenceClassification.from_pretrained(checkpoint_path)

        # copy from easynlp (start)
        model.eval()
        total_loss = 0
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        logits_list = list()
        y_trues = list()

        total_spent_time = 0.0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        for _step, batch in enumerate(valid_dataloader):
            try:
                batch = {
                    # key: val.cuda() if isinstance(val, torch.Tensor) else val
                    # for key, val in batch.items()
                    key:
                    val.to(device) if isinstance(val, torch.Tensor) else val
                    for key, val in batch.items()
                }
            except RuntimeError:
                batch = {key: val for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop('label_ids')
                outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert 'logits' in outputs
            logits = outputs['logits']

            y_trues.extend(label_ids.tolist())
            logits_list.extend(logits.tolist())
            hit_num += torch.sum(
                torch.argmax(logits, dim=-1) == label_ids).item()
            total_num += label_ids.shape[0]

            if len(logits.shape) == 1 or logits.shape[-1] == 1:
                tmp_loss = losses.mse_loss(logits, label_ids)
            elif len(logits.shape) == 2:
                tmp_loss = losses.cross_entropy(logits, label_ids)
            else:
                raise RuntimeError

            total_loss += tmp_loss.mean().item()
            total_steps += 1
            total_samples += valid_dataloader.batch_size
            if (_step + 1) % 100 == 0:
                total_step = len(
                    valid_dataloader.dataset) // valid_dataloader.batch_size
                logger.info('Eval: {}/{} steps finished'.format(
                    _step + 1, total_step))

        logger.info('Inference time = {:.2f}s, [{:.4f} ms / sample] '.format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info('Eval loss: {}'.format(eval_loss))

        logits_list = np.array(logits_list)
        eval_outputs = list()
        for metric in metrics[-1]:
            if metric.endswith('accuracy'):
                acc = hit_num / total_num
                logger.info('Accuracy: {}'.format(acc))
                eval_outputs.append(('accuracy', acc))
            elif metric == 'f1':
                if model.config.num_labels == 2:
                    f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1))
                    logger.info('F1: {}'.format(f1))
                    eval_outputs.append(('f1', f1))
                else:
                    f1 = f1_score(
                        y_trues,
                        np.argmax(logits_list, axis=-1),
                        average='macro')
                    logger.info('Macro F1: {}'.format(f1))
                    eval_outputs.append(('macro-f1', f1))
                    f1 = f1_score(
                        y_trues,
                        np.argmax(logits_list, axis=-1),
                        average='micro')
                    logger.info('Micro F1: {}'.format(f1))
                    eval_outputs.append(('micro-f1', f1))
            else:
                raise NotImplementedError('Metric %s not implemented' % metric)
        # copy from easynlp (end)

        return dict(eval_outputs)

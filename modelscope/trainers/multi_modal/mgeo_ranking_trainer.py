# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.models.nlp import BertForTextRanking
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp_trainer import NlpEpochBasedTrainer
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class GroupCollator():
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to
    List[qry], List[psg] and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def get_gis(self, gis, inps):
        gis_input_ids, gis_token_type_ids, gis_rel_type_ids = ([], [], [])
        gis_absolute_position_ids, gis_relative_position_ids = ([], [])
        gis_prov_ids, gis_city_ids, gis_dist_ids = ([], [], [])
        china_version = False
        for doc in inps:
            if len(doc) == 0:
                continue
            if len(doc[0]) == 6:
                for geom_id, geom_type, rel_type, absolute_position, relative_position, lxly in doc:
                    gis_input_ids.append(geom_id)
                    gis_token_type_ids.append(geom_type)
                    gis_rel_type_ids.append(rel_type)
                    gis_absolute_position_ids.append(absolute_position)
                    gis_relative_position_ids.append(relative_position)
            elif len(doc[0]) == 9:
                china_version = True
                for geom_id, geom_type, rel_type, absolute_position, relative_position, \
                        prov_id, city_id, dist_id, lxly in doc:
                    gis_input_ids.append(geom_id)
                    gis_token_type_ids.append(geom_type)
                    gis_rel_type_ids.append(rel_type)
                    gis_absolute_position_ids.append(absolute_position)
                    gis_relative_position_ids.append(relative_position)
                    gis_prov_ids.append(prov_id)
                    gis_city_ids.append(city_id)
                    gis_dist_ids.append(dist_id)

        gis.update(gis_input_ids, gis_token_type_ids, gis_rel_type_ids,
                   gis_absolute_position_ids, gis_relative_position_ids,
                   gis_prov_ids, gis_city_ids, gis_dist_ids, china_version)
        return gis

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(features[0], list):
            features = sum(features, [])
        keys = features[0].keys()
        batch = {k: list() for k in keys}
        for ele in features:
            for k, v in ele.items():
                batch[k].append(v)
        merged_batch = {}
        gis_list = []
        gis_tp = []
        for k in batch:
            if 'sentence1_gis' == k:
                gis = batch['gis1'][0]
                gis = self.get_gis(gis, batch['sentence1_gis'])
                if gis.prov_ids is not None:
                    gis_list.append({
                        'input_ids': gis.input_ids,
                        'attention_mask': gis.attention_mask,
                        'token_type_ids': gis.token_type_ids,
                        'rel_type_ids': gis.rel_type_ids,
                        'absolute_position_ids': gis.absolute_position_ids,
                        'relative_position_ids': gis.relative_position_ids,
                        'prov_ids': gis.prov_ids,
                        'city_ids': gis.city_ids,
                        'dist_ids': gis.dist_ids
                    })
                else:
                    gis_list.append({
                        'input_ids':
                        gis.input_ids,
                        'attention_mask':
                        gis.attention_mask,
                        'token_type_ids':
                        gis.token_type_ids,
                        'rel_type_ids':
                        gis.rel_type_ids,
                        'absolute_position_ids':
                        gis.absolute_position_ids,
                        'relative_position_ids':
                        gis.relative_position_ids
                    })
                gis_tp.append(torch.LongTensor([1]).to(gis.input_ids.device))
            elif 'sentence2_gis' == k:
                gis = batch['gis2'][0]
                gis = self.get_gis(gis, batch['sentence2_gis'])
                if gis.prov_ids is not None:
                    gis_list.append({
                        'input_ids': gis.input_ids,
                        'attention_mask': gis.attention_mask,
                        'token_type_ids': gis.token_type_ids,
                        'rel_type_ids': gis.rel_type_ids,
                        'absolute_position_ids': gis.absolute_position_ids,
                        'relative_position_ids': gis.relative_position_ids,
                        'prov_ids': gis.prov_ids,
                        'city_ids': gis.city_ids,
                        'dist_ids': gis.dist_ids
                    })
                else:
                    gis_list.append({
                        'input_ids':
                        gis.input_ids,
                        'attention_mask':
                        gis.attention_mask,
                        'token_type_ids':
                        gis.token_type_ids,
                        'rel_type_ids':
                        gis.rel_type_ids,
                        'absolute_position_ids':
                        gis.absolute_position_ids,
                        'relative_position_ids':
                        gis.relative_position_ids
                    })
                gis_tp.append(torch.LongTensor([0]).to(gis.input_ids.device))
            elif 'qid' in k or 'labels' in k:
                merged_batch[k] = torch.cat(batch[k], dim=0)
            elif not k.startswith('gis'):
                k_t = [it.t() for it in batch[k]]
                pad = torch.nn.utils.rnn.pad_sequence(k_t)
                if len(pad.size()) <= 2:
                    merged_batch[k] = pad.t()
                else:
                    l, b1, b2 = pad.size()
                    merged_batch[k] = pad.view(l, b1 * b2).t()
        if len(gis_list) > 0:
            merged_batch['gis_list'] = gis_list
        if len(gis_tp) > 0:
            merged_batch['gis_tp'] = gis_tp
        return merged_batch


@TRAINERS.register_module(module_name=Trainers.mgeo_ranking_trainer)
class MGeoRankingTrainer(NlpEpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Callable] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Preprocessor] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            **kwargs):

        if data_collator is None:
            data_collator = GroupCollator()

        super().__init__(
            model=model,
            cfg_file=cfg_file,
            cfg_modify_fn=cfg_modify_fn,
            arg_parse_fn=arg_parse_fn,
            data_collator=data_collator,
            preprocessor=preprocessor,
            optimizers=optimizers,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_revision=model_revision,
            **kwargs)

    def compute_mrr(self, result, k=10):
        mrr = 0
        for res in result.values():
            sorted_res = sorted(res, key=lambda x: x[0], reverse=True)
            ar = 0
            for index, ele in enumerate(sorted_res[:k]):
                if str(ele[1]) == '1':
                    ar = 1.0 / (index + 1)
                    break
            mrr += ar
        return mrr / len(result)

    def compute_ndcg(self, result, k=10):
        ndcg = 0
        from sklearn import ndcg_score
        for res in result.values():
            sorted_res = sorted(res, key=lambda x: [0], reverse=True)
            labels = np.array([[ele[1] for ele in sorted_res]])
            scores = np.array([[ele[0] for ele in sorted_res]])
            ndcg += float(ndcg_score(labels, scores, k=k))
        ndcg = ndcg / len(result)
        return ndcg

    def to_device(self, val, device):
        if isinstance(val, torch.Tensor):
            return val.to(device)
        elif isinstance(val, list):
            return [self.to_device(item, device) for item in val]
        elif isinstance(val, dict):
            new_val = {}
            for key in val:
                new_val[key] = self.to_device(val[key], device)
            return new_val
        print('can not convert to device')
        raise Exception('can not convert to device')

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        """evaluate a dataset

        evaluate a dataset via a specific model from the `checkpoint_path` path,
        if the `checkpoint_path` does not exist, read from the config file.

        Args:
            checkpoint_path (Optional[str], optional): the model path. Defaults
            to None.

        Returns:
            Dict[str, float]: the results about the evaluation Example:
            {"accuracy": 0.5091743119266054, "f1": 0.673780487804878}
        """
        # get the raw online dataset
        self.eval_dataloader = self._build_dataloader_with_dataset(
            self.eval_dataset,
            **self.cfg.evaluation.get('dataloader', {}),
            collate_fn=self.eval_data_collator)
        # generate a standard dataloader
        # generate a model
        if checkpoint_path is not None:
            model = BertForTextRanking.from_pretrained(checkpoint_path)
        else:
            model = self.model

        # copy from easynlp (start)
        model.eval()
        total_samples = 0

        logits_list = list()
        label_list = list()
        qid_list = list()

        total_spent_time = 0.0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        for _step, batch in enumerate(tqdm(self.eval_dataloader)):
            try:
                batch = self.to_device(batch, device)
            except RuntimeError:
                batch = {key: val for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop('labels').detach().cpu().numpy()
                qids = batch.pop('qid').detach().cpu().numpy()
                outputs = model(**batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time
            total_samples += self.eval_dataloader.batch_size

            def sigmoid(logits):
                return np.exp(logits) / (1 + np.exp(logits))

            logits = outputs['logits'].squeeze(-1).detach().cpu().numpy()
            logits = sigmoid(logits).tolist()

            label_list.extend(label_ids)
            logits_list.extend(logits)
            qid_list.extend(qids)

        logger.info('Inference time = {:.2f}s, [{:.4f} ms / sample] '.format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        rank_result = {}
        for qid, score, label in zip(qid_list, logits_list, label_list):
            if qid not in rank_result:
                rank_result[qid] = []
            rank_result[qid].append((score, label))

        for qid in rank_result:
            rank_result[qid] = sorted(rank_result[qid], key=lambda x: x[0])

        eval_outputs = list()
        for metric in self.metrics:
            if metric.startswith('mrr'):
                k = metric.split('@')[-1]
                k = int(k)
                mrr = self.compute_mrr(rank_result, k=k)
                logger.info('{}: {}'.format(metric, mrr))
                eval_outputs.append((metric, mrr))
            elif metric.startswith('ndcg'):
                k = metric.split('@')[-1]
                k = int(k)
                ndcg = self.compute_ndcg(rank_result, k=k)
                logger.info('{}: {}'.format(metric, ndcg))
                eval_outputs.append(('ndcg', ndcg))
            else:
                raise NotImplementedError('Metric %s not implemented' % metric)

        return dict(eval_outputs)

# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import TorchModel
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

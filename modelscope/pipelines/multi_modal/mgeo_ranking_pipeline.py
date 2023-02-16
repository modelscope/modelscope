# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['MGeoRankingPipeline']


@PIPELINES.register_module(
    Tasks.text_ranking, module_name=Pipelines.mgeo_ranking)
class MGeoRankingPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 sequence_length=128,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp word segment pipeline
           for prediction.

        Args:
            model (str or Model): Supply either a local model dir which
            supported the WS task, or a model id from the model hub, or a torch
            model instance. preprocessor (Preprocessor): An optional
            preprocessor instance, please make sure the preprocessor fits for
            the model if supplied. kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir,
                sequence_length=sequence_length,
                **kwargs)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the prediction results
        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: the predicted text representation
        """

        def sigmoid(logits):
            return np.exp(logits) / (1 + np.exp(logits))

        logits = inputs[OutputKeys.LOGITS].squeeze(-1).detach().cpu().numpy()
        pred_list = sigmoid(logits).tolist()
        return {OutputKeys.SCORES: pred_list}

    def get_gis(self, gis, inps):
        gis_input_ids, gis_token_type_ids, gis_rel_type_ids = ([], [], [])
        gis_absolute_position_ids, gis_relative_position_ids = ([], [])
        gis_prov_ids, gis_city_ids, gis_dist_ids = ([], [], [])
        china_version = False
        if len(inps[0]) == 6:
            for geom_id, geom_type, rel_type, absolute_position, relative_position, lxly in inps:
                gis_input_ids.append(geom_id)
                gis_token_type_ids.append(geom_type)
                gis_rel_type_ids.append(rel_type)
                gis_absolute_position_ids.append(absolute_position)
                gis_relative_position_ids.append(relative_position)
        elif len(inps[0]) == 9:
            china_version = True
            for geom_id, geom_type, rel_type, absolute_position, relative_position, \
                    prov_id, city_id, dist_id, lxly in inps:
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
        for att in vars(gis).keys():
            if isinstance(getattr(gis, att), torch.Tensor):
                setattr(gis, att, getattr(gis, att).to(self.device))
        return gis

    def _collate_fn(self, batch):
        merged_batch = {}
        gis_list = []
        gis_tp = []
        for k in batch:
            if 'sentence1_gis' == k:
                gis = batch['gis1']
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
                gis_tp.append(torch.LongTensor([1]).to(self.device))
            elif 'sentence2_gis' == k:
                gis = batch['gis2']
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
                gis_tp.append(torch.LongTensor([0]).to(self.device))
            elif 'qid' in k or 'labels' in k:
                merged_batch[k] = batch[k].to(self.device)
            elif not k.startswith('gis'):
                merged_batch[k] = batch[k].to(self.device)
        if len(gis_list) > 0:
            merged_batch['gis_list'] = gis_list
        if len(gis_tp) > 0:
            merged_batch['gis_tp'] = gis_tp
        return merged_batch

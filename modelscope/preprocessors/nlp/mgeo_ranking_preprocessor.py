# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch
from transformers import AutoTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert


class GisUtt:

    def __init__(self, pad_token_id, cls_token_id):
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None
        self.rel_type_ids = None
        self.absolute_position_ids = None
        self.relative_position_ids = None
        self.prov_ids = None
        self.city_ids = None
        self.dist_ids = None
        self.max_length = 32

    def update(self, gis_input_ids, gis_token_type_ids, gis_rel_type_ids,
               gis_absolute_position_ids, gis_relative_position_ids,
               gis_prov_ids, gis_city_ids, gis_dist_ids, china_version):
        gis_input_ids = [[self.cls_token_id] + f for f in gis_input_ids]
        gis_token_type_ids = [[self.pad_token_id] + f
                              for f in gis_token_type_ids]
        gis_rel_type_ids = [[self.pad_token_id] + f for f in gis_rel_type_ids]
        gis_absolute_position_ids = [[[self.pad_token_id] * 4] + f
                                     for f in gis_absolute_position_ids]
        gis_relative_position_ids = [[[self.pad_token_id] * 4] + f
                                     for f in gis_relative_position_ids]
        if china_version:
            gis_prov_ids = [[self.pad_token_id] + f for f in gis_prov_ids]
            gis_city_ids = [[self.pad_token_id] + f for f in gis_city_ids]
            gis_dist_ids = [[self.pad_token_id] + f for f in gis_dist_ids]

        gis_input_ids = [f[:self.max_length] for f in gis_input_ids]
        gis_token_type_ids = [f[:self.max_length] for f in gis_token_type_ids]
        gis_rel_type_ids = [f[:self.max_length] for f in gis_rel_type_ids]
        gis_absolute_position_ids = [
            f[:self.max_length] for f in gis_absolute_position_ids
        ]
        gis_relative_position_ids = [
            f[:self.max_length] for f in gis_relative_position_ids
        ]
        if china_version:
            gis_prov_ids = [f[:self.max_length] for f in gis_prov_ids]
            gis_city_ids = [f[:self.max_length] for f in gis_city_ids]
            gis_dist_ids = [f[:self.max_length] for f in gis_dist_ids]

        max_length = max([len(item) for item in gis_input_ids])
        self.input_ids = torch.tensor([
            f + [self.pad_token_id] * (max_length - len(f))
            for f in gis_input_ids
        ],
                                      dtype=torch.long)  # noqa: E126
        self.attention_mask = torch.tensor(
            [
                [1] * len(f) + [0] *  # noqa: W504
                (max_length - len(f)) for f in gis_input_ids
            ],
            dtype=torch.long)  # noqa: E126
        self.token_type_ids = torch.tensor([
            f + [self.pad_token_id] * (max_length - len(f))
            for f in gis_token_type_ids
        ],
                                           dtype=torch.long)  # noqa: E126
        self.rel_type_ids = torch.tensor([
            f + [self.pad_token_id] * (max_length - len(f))
            for f in gis_rel_type_ids
        ],
                                         dtype=torch.long)  # noqa: E126

        self.absolute_position_ids = torch.tensor(
            [
                f + [[self.pad_token_id] * 4] * (max_length - len(f))
                for f in gis_absolute_position_ids
            ],
            dtype=torch.long)  # noqa: E126
        self.relative_position_ids = torch.tensor(
            [
                f + [[self.pad_token_id] * 4] * (max_length - len(f))
                for f in gis_relative_position_ids
            ],
            dtype=torch.long)  # noqa: E126
        if china_version:
            self.prov_ids = torch.tensor([
                f + [self.pad_token_id] * (max_length - len(f))
                for f in gis_prov_ids
            ],
                                         dtype=torch.long)  # noqa: E126
            self.city_ids = torch.tensor([
                f + [self.pad_token_id] * (max_length - len(f))
                for f in gis_city_ids
            ],
                                         dtype=torch.long)  # noqa: E126
            self.dist_ids = torch.tensor([
                f + [self.pad_token_id] * (max_length - len(f))
                for f in gis_dist_ids
            ],
                                         dtype=torch.long)  # noqa: E126


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.mgeo_ranking)
class MGeoRankingTransformersPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 first_sequence='source_sentence',
                 second_sequence='sentences_to_compare',
                 first_sequence_gis='first_sequence_gis',
                 second_sequence_gis='second_sequence_gis',
                 label='labels',
                 qid='qid',
                 max_length=None,
                 **kwargs):
        """The tokenizer preprocessor class for the text ranking preprocessor.

        Args:
            model_dir(str, `optional`): The model dir used to parse the label mapping, can be None.
            first_sequence(str, `optional`): The key of the first sequence.
            second_sequence(str, `optional`): The key of the second sequence.
            label(str, `optional`): The keys of the label columns, default `labels`.
            qid(str, `optional`): The qid info.
            mode: The mode for the preprocessor.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
        """
        super().__init__(mode)
        self.model_dir = model_dir
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.first_sequence_gis = first_sequence_gis
        self.second_sequence_gis = second_sequence_gis

        self.label = label
        self.qid = qid
        self.sequence_length = max_length if max_length is not None else kwargs.get(
            'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    @type_assert(object, dict)
    def __call__(self,
                 data: Dict,
                 padding='longest',
                 truncation=True,
                 **kwargs) -> Dict[str, Any]:
        sentence1 = data.get(self.first_sequence)
        sentence2 = data.get(self.second_sequence)
        labels = data.get(self.label)
        qid = data.get(self.qid)
        sentence1_gis = data.get(self.first_sequence_gis)
        sentence2_gis = data.get(self.second_sequence_gis)
        if sentence1_gis is not None:
            sentence1_gis *= len(sentence2)

        if isinstance(sentence2, str):
            sentence2 = [sentence2]
        if isinstance(sentence1, str):
            sentence1 = [sentence1]
        sentence1 = sentence1 * len(sentence2)
        kwargs['max_length'] = kwargs.get(
            'max_length', kwargs.pop('sequence_length', self.sequence_length))
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        feature = self.tokenizer(
            sentence1,
            sentence2,
            padding=padding,
            truncation=truncation,
            **kwargs)
        if labels is not None:
            feature['labels'] = labels
        if qid is not None:
            feature['qid'] = qid
        if sentence1_gis is not None:
            feature['sentence1_gis'] = sentence1_gis
            gis = GisUtt(0, 1)
            feature['gis1'] = gis

        if sentence2_gis is not None:
            feature['sentence2_gis'] = sentence2_gis
            gis = GisUtt(0, 1)
            feature['gis2'] = gis

        return feature

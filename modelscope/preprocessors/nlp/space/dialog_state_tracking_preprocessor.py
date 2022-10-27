# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert
from .dst_processors import convert_examples_to_features, multiwoz22Processor

__all__ = ['DialogStateTrackingPreprocessor']


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.dialog_state_tracking_preprocessor)
class DialogStateTrackingPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

        from modelscope.models.nlp.space import SpaceConfig, SpaceTokenizer
        self.model_dir: str = model_dir
        self.config = SpaceConfig.from_pretrained(self.model_dir)
        self.tokenizer = SpaceTokenizer.from_pretrained(self.model_dir)
        self.processor = multiwoz22Processor()

    @type_assert(object, dict)
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (Dict[str, Any]): a sentence
                Example:
                    {
                        'utter': {'User-1': "Hi, I'm looking for a train that is going"
                            "to cambridge and arriving there by 20:45, is there anything like that?"},
                        'history_states': [{}]
                    }

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        import torch
        from torch.utils.data import (DataLoader, RandomSampler,
                                      SequentialSampler)

        utter = data['utter']
        history_states = data['history_states']
        example = self.processor.create_example(
            inputs=utter,
            history_states=history_states,
            set_type='test',
            slot_list=self.config.dst_slot_list,
            label_maps={},
            append_history=True,
            use_history_labels=True,
            swap_utterances=True,
            label_value_repetitions=True,
            delexicalize_sys_utts=True,
            unk_token='[UNK]',
            analyze=False)

        features = convert_examples_to_features(
            examples=[example],
            slot_list=self.config.dst_slot_list,
            class_types=self.config.dst_class_types,
            model_type=self.config.model_type,
            tokenizer=self.tokenizer,
            max_seq_length=180,  # args.max_seq_length
            slot_value_dropout=(0.0))

        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                       dtype=torch.long)
        all_example_index = torch.arange(
            all_input_ids.size(0), dtype=torch.long)
        f_start_pos = [f.start_pos for f in features]
        f_end_pos = [f.end_pos for f in features]
        f_inform_slot_ids = [f.inform_slot for f in features]
        f_refer_ids = [f.refer_id for f in features]
        f_diag_state = [f.diag_state for f in features]
        f_class_label_ids = [f.class_label_id for f in features]
        all_start_positions = {}
        all_end_positions = {}
        all_inform_slot_ids = {}
        all_refer_ids = {}
        all_diag_state = {}
        all_class_label_ids = {}
        for s in self.config.dst_slot_list:
            all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos],
                                                  dtype=torch.long)
            all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos],
                                                dtype=torch.long)
            all_inform_slot_ids[s] = torch.tensor(
                [f[s] for f in f_inform_slot_ids], dtype=torch.long)
            all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids],
                                            dtype=torch.long)
            all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state],
                                             dtype=torch.long)
            all_class_label_ids[s] = torch.tensor(
                [f[s] for f in f_class_label_ids], dtype=torch.long)
        dataset = [
            all_input_ids, all_input_mask, all_segment_ids,
            all_start_positions, all_end_positions, all_inform_slot_ids,
            all_refer_ids, all_diag_state, all_class_label_ids,
            all_example_index
        ]

        with torch.no_grad():
            diag_state = {
                slot:
                torch.tensor([0 for _ in range(self.config.eval_batch_size)
                              ]).to(self.config.device)
                for slot in self.config.dst_slot_list
            }

        if len(history_states) > 2:
            ds = history_states[-2]
        else:
            ds = {slot: 'none' for slot in self.config.dst_slot_list}

        return {
            'batch': dataset,
            'features': features,
            'diag_state': diag_state,
            'ds': ds
        }

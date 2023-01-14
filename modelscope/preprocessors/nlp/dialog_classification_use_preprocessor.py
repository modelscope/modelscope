# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.hub import parse_label_mapping


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.dialog_use_preprocessor)
class DialogueClassificationUsePreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 label2id: Dict = None,
                 max_length: int = None):
        """The preprocessor for user satisfaction estimation task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir containing the essential files to build the tokenizer.
            label2id: The dict with label-id mappings, default the label_mapping.json file in the model_dir.
            max_length: The max length of dialogue, default 30.
        """
        super().__init__()
        self.model_dir: str = model_dir
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        self.max_seq_len = min(max_length,
                               30) if max_length is not None else 30
        self.label2id = label2id
        if self.label2id is None and self.model_dir is not None:
            self.label2id = parse_label_mapping(self.model_dir)

    @property
    def id2label(self):
        """Return the id2label mapping according to the label2id mapping.

        @return: The id2label mapping if exists.
        """
        if self.label2id is not None:
            return {id: label for label, id in self.label2id.items()}
        return None

    def __call__(self, data: List[Tuple[str]]) -> Dict[str, Any]:
        input_ids = []
        for pair in data:
            ids = []
            for sent in str(pair).split('|||'):
                ids += self.tokenizer.encode(sent)[1:]
                if len(ids) >= self.max_seq_len - 1:
                    ids = ids[:self.max_seq_len - 2] + [102]
                    break
            input_ids.append([101] + ids)  # [CLS] + (max_len-1) tokens
        input_ids = [torch.tensor(utt, dtype=torch.long) for utt in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids = input_ids.view(1, len(data), -1)
        rst = {'input_ids': input_ids}
        return rst

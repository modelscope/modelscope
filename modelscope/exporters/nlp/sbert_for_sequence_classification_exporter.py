import os
from collections import OrderedDict
from typing import Any, Dict, Mapping, Tuple

from torch.utils.data.dataloader import default_collate

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.torch_model_exporter import TorchModelExporter
from modelscope.metainfo import Models
from modelscope.preprocessors import (
    Preprocessor, TextClassificationTransformersPreprocessor,
    build_preprocessor)
from modelscope.utils.constant import ModeKeys, Tasks


@EXPORTERS.register_module(Tasks.text_classification, module_name=Models.bert)
@EXPORTERS.register_module(
    Tasks.text_classification, module_name=Models.structbert)
@EXPORTERS.register_module(Tasks.sentence_similarity, module_name=Models.bert)
@EXPORTERS.register_module(
    Tasks.zero_shot_classification, module_name=Models.bert)
@EXPORTERS.register_module(
    Tasks.sentiment_classification, module_name=Models.bert)
@EXPORTERS.register_module(Tasks.nli, module_name=Models.bert)
@EXPORTERS.register_module(
    Tasks.sentence_similarity, module_name=Models.structbert)
@EXPORTERS.register_module(
    Tasks.sentiment_classification, module_name=Models.structbert)
@EXPORTERS.register_module(Tasks.nli, module_name=Models.structbert)
@EXPORTERS.register_module(
    Tasks.zero_shot_classification, module_name=Models.structbert)
class SbertForSequenceClassificationExporter(TorchModelExporter):

    def generate_dummy_inputs(self,
                              shape: Tuple = None,
                              pair: bool = False,
                              **kwargs) -> Dict[str, Any]:
        """Generate dummy inputs for model exportation to onnx or other formats by tracing.

        Args:
            shape: A tuple of input shape which should have at most two dimensions.
                shape = (1, ) batch_size=1, sequence_length will be taken from the preprocessor.
                shape = (8, 128) batch_size=1, sequence_length=128, which will cover the config of the preprocessor.
            pair(bool, `optional`): Whether to generate sentence pairs or single sentences.

        Returns:
            Dummy inputs.
        """

        assert hasattr(
            self.model, 'model_dir'
        ), 'model_dir attribute is required to build the preprocessor'
        batch_size = 1
        sequence_length = {}
        if shape is not None:
            if len(shape) == 1:
                batch_size = shape[0]
            elif len(shape) == 2:
                batch_size, max_length = shape
                sequence_length = {'sequence_length': max_length}

        preprocessor = Preprocessor.from_pretrained(
            self.model.model_dir,
            preprocessor_mode=ModeKeys.TRAIN,
            task=Tasks.text_classification,
            **sequence_length)
        if pair:
            first_sequence = preprocessor.nlp_tokenizer.tokenizer.unk_token
            second_sequence = preprocessor.nlp_tokenizer.tokenizer.unk_token
        else:
            first_sequence = preprocessor.nlp_tokenizer.tokenizer.unk_token
            second_sequence = None

        batched = []
        for _ in range(batch_size):
            batched.append(preprocessor((first_sequence, second_sequence)))
        return default_collate(batched)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        dynamic_axis = {0: 'batch', 1: 'sequence'}
        return OrderedDict([
            ('input_ids', dynamic_axis),
            ('attention_mask', dynamic_axis),
            ('token_type_ids', dynamic_axis),
        ])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict({'logits': {0: 'batch'}})

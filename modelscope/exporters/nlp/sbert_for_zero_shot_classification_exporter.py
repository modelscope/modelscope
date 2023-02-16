from collections import OrderedDict
from typing import Any, Dict, Mapping

from modelscope.exporters.builder import EXPORTERS
from modelscope.exporters.torch_model_exporter import TorchModelExporter
from modelscope.metainfo import Models
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks


@EXPORTERS.register_module(
    Tasks.zero_shot_classification, module_name=Models.bert)
@EXPORTERS.register_module(
    Tasks.zero_shot_classification, module_name=Models.structbert)
class SbertForZeroShotClassificationExporter(TorchModelExporter):

    def generate_dummy_inputs(self,
                              candidate_labels,
                              hypothesis_template,
                              max_length=128,
                              pair: bool = False,
                              **kwargs) -> Dict[str, Any]:
        """Generate dummy inputs for model exportation to onnx or other formats by tracing.

        Args:

            max_length(int): The max length of sentence, default 128.
            hypothesis_template(str): The template of prompt, like '这篇文章的标题是{}'
            candidate_labels(List): The labels of prompt,
            like ['文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事']
            pair(bool, `optional`): Whether to generate sentence pairs or single sentences.

        Returns:
            Dummy inputs.
        """

        assert hasattr(
            self.model, 'model_dir'
        ), 'model_dir attribute is required to build the preprocessor'
        preprocessor = Preprocessor.from_pretrained(
            self.model.model_dir, max_length=max_length)
        return preprocessor(
            preprocessor.nlp_tokenizer.tokenizer.unk_token,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template)

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

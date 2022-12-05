# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.nlp.plug import DistributedPlug
from modelscope.pipelines.base import DistributedPipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TextGenerationTransformersPreprocessor
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_generation, module_name=Pipelines.plug_generation)
class DistributedPlugPipeline(DistributedPipeline):
    """This class is used to instantiate the plug model.
    """

    model = None

    def __init__(self,
                 model,
                 preprocessor=None,
                 first_sequence='sentence',
                 sequence_length=512,
                 **kwargs):
        """Create a plug pipeline instance.

        Args:
        model: The model_id of plug(damo/nlp_plug_text-generation_27B).
        The default path to damo/nlp_plug_text-generation_27B can be obtained by function
        get_cache_dir("damo/nlp_plug_text-generation_27B"), the model should be downloaded to
        this path before calling this class by model_id.
        The model can be downloaded from the link on
        https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary.
        After downloading, you should have a plug model structure like this:
        /your/path/to/damo/nlp_plug_text-generation_27B
            |_ config.json
            |_ configuration.json
            |_ ds_zero-offload_10B_config.json
            |_ vocab.txt
            |_ model <-- an empty directory

        Model binaries shall be downloaded separately to populate the model directory, so that
        the model directory would contain the following binaries:
            |_ model
                |_ mp_rank_00_model_states.pt
                |_ mp_rank_01_model_states.pt
                |_ mp_rank_02_model_states.pt
                |_ mp_rank_03_model_states.pt
                |_ mp_rank_04_model_states.pt
                |_ mp_rank_05_model_states.pt
                |_ mp_rank_06_model_states.pt
                |_ mp_rank_07_model_states.pt
        preprocessor: The optional preprocessor, if not passed in, a TextGenerationPreprocessor will
            be used as default.
        kwargs (dict, `optional`): Extra kwargs passed into the preprocessor's constructor.
        """
        if preprocessor is None:
            preprocessor = TextGenerationTransformersPreprocessor(
                model,
                first_sequence=first_sequence,
                sequence_length=sequence_length,
                **kwargs)
        super().__init__(model, preprocessor=preprocessor, **kwargs)
        self.cls_token_id = preprocessor.nlp_tokenizer.tokenizer.cls_token_id

    @classmethod
    def _forward_one(cls, inputs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return cls.model.generate(inputs['inputs'],
                                      **inputs['forward_params'])

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        batch_size = inputs['input_ids'].shape[0]
        dec_input_ids = torch.full([batch_size, 1],
                                   self.cls_token_id,
                                   dtype=torch.long)
        inputs['dec_input_ids'] = dec_input_ids
        res = super().forward(inputs, **forward_params)
        return res

    @classmethod
    def _instantiate_one(cls, rank, model_dir, **kwargs):
        cls.model = DistributedPlug(model_dir, rank, **kwargs)
        cls.model.eval()

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        from modelscope.outputs import OutputKeys
        generate_context = inputs['generate_context']
        generate_context = ''.join(
            self.preprocessor.nlp_tokenizer.tokenizer.convert_ids_to_tokens(
                generate_context)).replace('[UNK]', '“').replace('##', '')
        return {OutputKeys.TEXT: generate_context}

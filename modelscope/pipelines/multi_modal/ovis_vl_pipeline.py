import torch
from typing import Any, Dict, Union
from PIL import Image

from modelscope import AutoModelForCausalLM
from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Frameworks, Fields, Tasks
from modelscope.pipelines.multi_modal.visual_question_answering_pipeline import VisualQuestionAnsweringPipeline


@PIPELINES.register_module(
    Tasks.visual_question_answering, module_name="ovis-vl")
class VisionChatPipeline(VisualQuestionAnsweringPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        # super().__init__
        self.device_name = device
        self.framework = Frameworks.torch
        self._model_prepare = True
        self._auto_collate = auto_collate

        # ovis
        self.device = 'cuda' if device == 'gpu' else device
        self.model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                     multimodal_max_length=8192,
                                                     trust_remote_code=True).to(self.device)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()


    def preprocess(self, inputs: Dict[str, Any]):
        text = inputs['text']
        image = inputs['image']
        image = Image.open(image)
        query = f'<image>\n{text}'
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        return {'input_ids':input_ids, 'pixel_values': pixel_values, 'attention_mask': attention_mask}
      
    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        input_ids = inputs['input_ids']
        pixel_values = inputs['pixel_values']
        attention_mask = inputs['attention_mask']

        max_new_tokens = forward_params.get("max_new_tokens", 1024)
        do_sample = forward_params.get("do_sample", False)
        top_p = forward_params.get("top_p", None)
        top_k = forward_params.get("top_k", None)
        temperature = forward_params.get("temperature", None)
        repetition_penalty = forward_params.get("repetition_penalty", None)
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        return {'output_ids': output_ids}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output_ids = inputs['output_ids']
        output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return {OutputKeys.TEXT: output}

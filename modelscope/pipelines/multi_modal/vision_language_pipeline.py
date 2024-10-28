import torch
from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys, AwesomeTaskOutput
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Fields, Tasks


# Pipeline按照任务名称+pipeline名字进行注册。configuration.json中只要添加pipeline.type字段即可使用，不需要改动代码
@PIPELINES.register_module(
    Tasks.ovis_vision_chat, module_name=Pipelines.ovis_vision_chat)
class VisionChatPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            **kwargs)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()


    def preprocess(self, inputs: Dict[str, Any]):
        text = inputs['text']
        image = inputs['image']
        query = f'<image>\n{text}'
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        return {'input_ids':input_ids, 'pixel_values': pixel_values, 'attention_mask': attention_mask}
      
    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Union[Dict[str, Any], AwesomeTaskOutput]:
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
        return output_ids

    def postprocess(self,
                    inputs: Union[Dict[str, Any],
                                  AwesomeTaskOutput]) -> Dict[str, Any]:
        # do some post-processes
        output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return {OutputKeys.TEXT: output}
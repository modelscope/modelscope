# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys, TokenGeneratorOutput
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks

context_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}'
history_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                   '#以下是在此之前我们的对话内容，可作为回复时的参考。{history}'
knowledge_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                     '#以下是和对话相关的知识，请你参考该知识进行回复。{knowledge}'
user_profile_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                        '#假设以下是你对我所了解的信息，请你参考该信息并避免你的回复和该信息矛盾，信息如下：{user_profile}'
bot_profile_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                       '#假设以下是你的人物设定，请你参考该信息并避免你的回复和该信息矛盾，信息如下：{bot_profile}'

__all__ = ['FidDialoguePipeline']


@PIPELINES.register_module(
    Tasks.fid_dialogue, module_name=Pipelines.fid_dialogue)
class FidDialoguePipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """Use `model` and `preprocessor` to create a fid-dialogue pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the text generation task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
            Examples:
                >>> from modelscope.pipelines import pipeline
                >>> from modelscope.utils.constant import Tasks
                >>> pipeline_ins = pipeline(Tasks.fid_dialogue, model='damo/plug-dialogue', model_revision='v1.0.1')
                >>> input = {
                >>>    "history": "你好[SEP]你好，我是小达，很高兴认识你！[SEP]李白是谁",
                >>>    "bot_profile": "我是小达;我是女生;我是单身;我今年21岁;我生日是2001年11月11日",
                >>>    "knowledge": "唐代诗人李白（701年—762年12月）,字太白,号青莲居士,又号“谪仙人”[SEP]李白（公元701年—公元762年），字太白",
                >>>    "user_profile": "你是小明"
                >>> }
                >>> result = pipeline_ins(input)
                >>> print(result)
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            **kwargs)

        if preprocessor is None:
            self.preprocessor_tokenizer = Preprocessor.from_pretrained(
                self.model.model_dir, **kwargs)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        self.model = self.model.to(self.device)
        self.model.eval()

        self.SEP = '[SEP]'

    def forward(self, inputs: Dict[str, Any], **forward_params):
        with torch.no_grad():
            return self.model.generate(inputs, **forward_params)

    def preprocess(self, inputs: Dict[str, Any],
                   **preprocess_params) -> Dict[str, Any]:
        # init params
        max_encoder_length = 300
        if 'max_encoder_length' in preprocess_params:
            max_encoder_length = preprocess_params.pop('max_encoder_length')
        # get raw data
        history = inputs['history'] if 'history' in inputs else ''
        if len(history) <= 0:
            raise Exception('history is necessary!')
        knowledge = inputs['knowledge'] if 'knowledge' in inputs else ''
        user_profile = inputs[
            'user_profile'] if 'user_profile' in inputs else ''
        bot_profile = inputs['bot_profile'] if 'bot_profile' in inputs else ''
        # parse raw data
        history = history.split(self.SEP)
        context = history[-3:]
        context = self.process_context(context)
        history = history[:-3]
        history = self.process_history(history)
        knowledge = knowledge.split(self.SEP)

        model_input = []
        if history and len(history) > 0:
            model_input.append(
                history_template.format(context=context, history=history))
        if knowledge and len(knowledge) > 0:
            for know in knowledge:
                model_input.append(
                    knowledge_template.format(context=context, knowledge=know))
        if user_profile and len(user_profile) > 0:
            model_input.append(
                user_profile_template.format(
                    context=context, user_profile=user_profile))
        if bot_profile and len(bot_profile) > 0:
            model_input.append(
                bot_profile_template.format(
                    context=context, bot_profile=bot_profile))

        if not model_input:
            model_input.append(context_template.format(context=context))

        for i in range(len(model_input)):
            model_input[i] = re.sub('[ \t]+', '▂', model_input[i])

        # tokenization
        input_ids = self.preprocessor_tokenizer(
            {'src_txt': model_input},
            padding=True,
            truncation=True,
            max_length=max_encoder_length,
            return_tensors='pt')['input_ids'].unsqueeze(0).to(self.device)
        input_dict = {
            'input_ids':
            input_ids.to(torch.int64).to(self.device),
            'attention_mask': (input_ids != 0).to(torch.int64).to(self.device),
            'token_type_ids':
            torch.zeros(input_ids.shape).to(torch.int64).to(self.device)
        }

        return input_dict

    def process_context(self, context_list):
        subject = '我'
        for i in range(len(context_list) - 1, -1, -1):
            if len(context_list[i]) > 0 and context_list[i][
                    -1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
                context_list[i] = context_list[i] + '。'
            context_list[i] = subject + '：' + context_list[i]
            subject = '你' if subject == '我' else '我'
        return ''.join(context_list)

    def process_history(self, history_list):
        subject = '你'
        for i in range(len(history_list) - 1, -1, -1):
            if len(history_list[i]) > 0 and history_list[i][
                    -1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
                history_list[i] = history_list[i] + '。'
            history_list[i] = subject + '：' + history_list[i]
            subject = '你' if subject == '我' else '我'
        return ''.join(history_list)

    def postprocess(self, inputs: TokenGeneratorOutput,
                    **postprocess_params) -> Dict[str, Any]:

        if torch.cuda.is_available():
            hypotheses = inputs.sequences.detach().cpu().tolist()

        response = self.preprocessor_tokenizer.decode(
            hypotheses[0], skip_special_tokens=True)
        response = response.replace(' ', '')
        return {OutputKeys.TEXT: response}

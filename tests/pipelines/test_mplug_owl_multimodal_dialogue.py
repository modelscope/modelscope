# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from PIL import Image

from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MplugOwlMultimodalDialogueTest(unittest.TestCase):

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_multimodal_dialogue_with_model(self):
        model = Model.from_pretrained(
            'damo/multi-modal_mplug_owl_multimodal-dialogue_7b')
        pipeline_multimodal_dialogue = pipeline(
            task=Tasks.multimodal_dialogue, model=model)
        image = 'data/resource/portrait_input.png'
        system_prompt_1 = 'The following is a conversation between a curious human and AI assistant.'
        system_prompt_2 = "The assistant gives helpful, detailed, and polite answers to the user's questions."
        messages = {
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt_1 + ' ' + system_prompt_2
                },
                {
                    'role': 'user',
                    'content': [{
                        'image': image
                    }]
                },
                {
                    'role': 'user',
                    'content': 'Describe the facial expression of the man.'
                },
            ]
        }
        result = pipeline_multimodal_dialogue(messages)
        print(result[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_multimodal_dialogue_with_name(self):
        pipeline_multimodal_dialogue = pipeline(
            Tasks.multimodal_dialogue,
            model='damo/multi-modal_mplug_owl_multimodal-dialogue_7b')
        image = 'data/resource/portrait_input.png'
        system_prompt_1 = 'The following is a conversation between a curious human and AI assistant.'
        system_prompt_2 = "The assistant gives helpful, detailed, and polite answers to the user's questions."
        messages = {
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt_1 + ' ' + system_prompt_2
                },
                {
                    'role': 'user',
                    'content': [{
                        'image': image
                    }]
                },
                {
                    'role': 'user',
                    'content': 'Describe the facial expression of the man.'
                },
            ]
        }
        result = pipeline_multimodal_dialogue(messages, max_new_tokens=512)
        print(result[OutputKeys.TEXT])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_multimodal_dialogue_with_text(self):
        pipeline_multimodal_dialogue = pipeline(
            Tasks.multimodal_dialogue,
            model='damo/multi-modal_mplug_owl_multimodal-dialogue_7b')
        system_prompt_1 = 'The following is a conversation between a curious human and AI assistant.'
        system_prompt_2 = "The assistant gives helpful, detailed, and polite answers to the user's questions."
        messages = {
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt_1 + ' ' + system_prompt_2
                },
                {
                    'role': 'user',
                    'content': 'Where is the captial of China?'
                },
            ]
        }
        result = pipeline_multimodal_dialogue(messages, max_new_tokens=512)
        print(result[OutputKeys.TEXT])


if __name__ == '__main__':
    unittest.main()

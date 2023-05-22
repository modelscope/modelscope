import base64
import unittest

import json

from modelscope.utils.constant import Tasks
from modelscope.utils.input_output import (
    PipelineInfomation, service_base64_input_to_pipeline_input)


def encode_image_to_base64(image):
    base64_str = str(base64.b64encode(image), 'utf-8')
    return base64_str


class PipelineInputOutputTest(unittest.TestCase):

    def test_template_pipeline_dict_input(self):
        pipeline_info = PipelineInfomation(
            Tasks.task_template, 'PipelineTemplate',
            'modelscope/pipelines/pipeline_template.py')
        schema = pipeline_info.schema
        expect_schema = {
            'input': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'description':
                        'Base64 encoded image file or url string.'
                    },
                    'text': {
                        'type': 'string',
                        'description': 'The input text.'
                    }
                }
            },
            'parameters': {
                'type': 'object',
                'properties': {
                    'max_length': {
                        'type': 'integer',
                        'default': 1024
                    },
                    'top_p': {
                        'type': 'number',
                        'default': 0.8
                    },
                    'postprocess_param1': {
                        'type': 'string',
                        'default': None
                    }
                }
            },
            'output': {
                'type': 'object',
                'properties': {
                    'boxes': {
                        'type': 'array',
                        'items': {
                            'type': 'number'
                        }
                    },
                    'output_img': {
                        'type': 'string',
                        'description': 'The base64 encoded image.'
                    },
                    'text_embedding': {
                        'type': 'array',
                        'items': {
                            'type': 'number'
                        }
                    }
                }
            }
        }
        assert expect_schema == schema

    def test_template_pipeline_list_input(self):
        pipeline_info = PipelineInfomation(
            Tasks.text_classification, 'LanguageIdentificationPipeline',
            'modelscope/pipelines/nlp/language_identification_pipline.py')
        schema = pipeline_info.schema
        expect_schema = {
            'input': {
                'type': 'object',
                'properties': {
                    'text': {
                        'type': 'string',
                        'description': 'The input text.'
                    },
                    'text2': {
                        'type': 'string',
                        'description': 'The input text.'
                    }
                }
            },
            'parameters': {},
            'output': {
                'type': 'object',
                'properties': {
                    'scores': {
                        'type': 'array',
                        'items': {
                            'type': 'number'
                        }
                    },
                    'labels': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    }
                }
            }
        }
        assert expect_schema == schema

    def test_input_output_encode_decode(self):
        with open('data/test/images/image_captioning.png', 'rb') as f:
            image = f.read()
        text = 'hello schema.'
        request_json = {
            'input': {
                'image': encode_image_to_base64(image),
                'text': text
            },
            'parameters': {
                'max_length': 10000,
                'top_p': 0.8
            }
        }
        pipeline_inputs, parameters = service_base64_input_to_pipeline_input(
            Tasks.task_template, request_json)
        assert 'image' in pipeline_inputs
        assert pipeline_inputs['text'] == text
        assert parameters['max_length'] == 10000
        assert parameters['top_p'] == 0.8


if __name__ == '__main__':
    unittest.main()

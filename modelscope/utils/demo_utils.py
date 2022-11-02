# Copyright (c) Alibaba, Inc. and its affiliates.

import io

import cv2
import json

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks, TasksIODescriptions
from modelscope.utils.service_utils import NumpyEncoder

TASKS_INPUT_TEMPLATES = {
    # vision tasks
    Tasks.image_portrait_stylization: TasksIODescriptions.image_to_image,
    Tasks.portrait_matting: TasksIODescriptions.image_to_image,
    Tasks.skin_retouching: TasksIODescriptions.image_to_image,
    Tasks.image_captioning: TasksIODescriptions.image_to_text,
    Tasks.image_denoising: TasksIODescriptions.image_to_image,
    Tasks.image_portrait_enhancement: TasksIODescriptions.image_to_image,
    Tasks.image_super_resolution: TasksIODescriptions.image_to_image,
    Tasks.image_colorization: TasksIODescriptions.image_to_image,
    Tasks.image_color_enhancement: TasksIODescriptions.image_to_image,
    Tasks.face_image_generation: TasksIODescriptions.seed_to_image,
    Tasks.image_style_transfer: TasksIODescriptions.images_to_image,
    Tasks.image_segmentation: TasksIODescriptions.image_to_text,
    Tasks.image_object_detection: TasksIODescriptions.image_to_text,

    # not tested
    Tasks.image_classification: TasksIODescriptions.image_to_text,
    Tasks.ocr_detection: TasksIODescriptions.image_to_text,
    Tasks.ocr_recognition: TasksIODescriptions.image_to_text,
    Tasks.body_2d_keypoints: TasksIODescriptions.image_to_text,

    # nlp tasks
    Tasks.text_classification: TasksIODescriptions.text_to_text,
    Tasks.text_generation: TasksIODescriptions.text_to_text,
    Tasks.word_segmentation: TasksIODescriptions.text_to_text,
    Tasks.text_error_correction: TasksIODescriptions.text_to_text,
    Tasks.named_entity_recognition: TasksIODescriptions.text_to_text,
    Tasks.sentiment_classification: TasksIODescriptions.text_to_text,

    # audio tasks
    Tasks.text_to_speech: TasksIODescriptions.text_to_speech,
    Tasks.auto_speech_recognition: TasksIODescriptions.speech_to_text,
    Tasks.keyword_spotting: TasksIODescriptions.speech_to_text,
    Tasks.acoustic_noise_suppression: TasksIODescriptions.speech_to_speech,
    Tasks.acoustic_echo_cancellation: TasksIODescriptions.speeches_to_speech,

    # multi-modal
    Tasks.visual_grounding: TasksIODescriptions.visual_grounding,
    Tasks.visual_question_answering:
    TasksIODescriptions.visual_question_answering,
    Tasks.visual_entailment: TasksIODescriptions.visual_entailment,
    Tasks.generative_multi_modal_embedding:
    TasksIODescriptions.generative_multi_modal_embedding,

    # new tasks
    Tasks.virtual_try_on: TasksIODescriptions.images_to_image,

    # TODO(lingcai.wl): support more tasks and implement corresponding example
}

INPUT_EXAMPLES = {
    # Must align with task schema defined in the Widget section of model card=
    # cv
    TasksIODescriptions.image_to_image: {
        'inputs': [
            'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png'
        ],
        'urlPaths': {
            'outUrls': [{
                'outputKey': OutputKeys.OUTPUT_IMG,
                'fileType': 'png'
            }]
        }
    },
    TasksIODescriptions.images_to_image: {
        'inputs': [
            'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_content.jpg',
            'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_style.jpg'
        ],
        'urlPaths': {
            'outUrls': [{
                'outputKey': OutputKeys.OUTPUT_IMG,
                'fileType': 'png'
            }]
        }
    },
    TasksIODescriptions.image_to_text: {
        'inputs': [
            'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png'
        ],
        'urlPaths': {}
    },
    # nlp
    TasksIODescriptions.text_to_text: {
        'inputs': ['test'],
        'urlPaths': {}
    },

    # audio
    TasksIODescriptions.speech_to_text: {
        'inputs': [
            'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav'
        ],
        'urlPaths': {}
    },
    TasksIODescriptions.text_to_speech: {
        'inputs': ['北京今天天气怎么样'],
        'urlPaths': {
            'outUrls': [{
                'outputKey': OutputKeys.OUTPUT_PCM,
                'fileType': 'pcm'
            }]
        }
    },
    TasksIODescriptions.speeches_to_speech: {
        'inputs': [
            'http://225252-file.oss-cn-hangzhou-zmf.aliyuncs.com/maas_demo/nearend_mic.wav',
            'http://225252-file.oss-cn-hangzhou-zmf.aliyuncs.com/maas_demo/nearend_speech.wav'
        ],
        'urlPaths': {
            'outUrls': [{
                'outputKey': OutputKeys.OUTPUT_PCM,
                'fileType': 'pcm'
            }]
        }
    },
    TasksIODescriptions.speech_to_speech: {
        'inputs': [
            'http://225252-file.oss-cn-hangzhou-zmf.aliyuncs.com/maas_demo/speech_with_noise.wav'
        ],
        'urlPaths': {
            'outUrls': [{
                'outputKey': OutputKeys.OUTPUT_PCM,
                'fileType': 'pcm'
            }]
        }
    },

    # multi modal
    TasksIODescriptions.visual_grounding: {
        'task':
        Tasks.visual_grounding,
        'inputs': [
            'http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-grounding/visual_grounding.png',
            'a blue turtle-like pokemon with round head'
        ],
        'urlPaths': {
            'inUrls': [{
                'name': 'image'
            }, {
                'name': 'text'
            }]
        }
    },
    TasksIODescriptions.visual_question_answering: {
        'task':
        Tasks.visual_question_answering,
        'inputs': [
            'http://225252-file.oss-cn-hangzhou-zmf.aliyuncs.com/maas_demo/visual_question_answering.png',
            'what is grown on the plant?'
        ],
        'urlPaths': {
            'inUrls': [{
                'name': 'image'
            }, {
                'name': 'text'
            }],
            'outUrls': [{
                'outputKey': 'text'
            }]
        }
    },
    TasksIODescriptions.visual_entailment: {
        'task':
        Tasks.visual_entailment,
        'inputs': [
            'http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-entailment/visual_entailment.jpg',
            'there are two birds.', 'test'
        ],
        'urlPaths': {
            'inUrls': [{
                'name': 'image'
            }, {
                'name': 'text'
            }],
            'outUrls': [{}]
        }
    },
    TasksIODescriptions.generative_multi_modal_embedding: {
        'task':
        Tasks.generative_multi_modal_embedding,
        'inputs': [
            'http://clip-multimodal.oss-cn-beijing.aliyuncs.com/lingchen/demo/dogs.jpg',
            'dogs playing in the grass'
        ],
        'urlPaths': {
            'inUrls': [{
                'name': 'image'
            }, {
                'name': 'text'
            }],
            'outUrls': [{}]
        }
    },
}


class DemoCompatibilityCheck(object):

    def compatibility_check(self):
        if self.task not in TASKS_INPUT_TEMPLATES:
            print('task is not supported in demo service so far')
            return False
        if TASKS_INPUT_TEMPLATES[self.task] not in INPUT_EXAMPLES:
            print('no example input for this task')
            return False

        print('testing demo: ', self.task, self.model_id)
        test_pipline = pipeline(self.task, self.model_id)
        req = INPUT_EXAMPLES[TASKS_INPUT_TEMPLATES[self.task]]
        inputs = preprocess(req)
        params = req.get('parameters', {})
        # modelscope inference
        if params != {}:
            output = test_pipline(inputs, **params)
        else:
            output = test_pipline(inputs)
        json.dumps(output, cls=NumpyEncoder)
        result = postprocess(req, output)
        print(result)
        return True


def preprocess(req):
    in_urls = req.get('urlPaths').get('inUrls')
    if len(req['inputs']) == 1:
        inputs = req['inputs'][0]
    else:
        inputs = tuple(req['inputs'])
    if in_urls is None or len(in_urls) == 0:
        return inputs

    inputs_dict = {}
    for i, in_url in enumerate(in_urls):
        input_name = in_url.get('name')
        if input_name is None or input_name == '':
            return inputs
        inputs_dict[input_name] = req['inputs'][i]
    return inputs_dict


def postprocess(req, resp):
    out_urls = req.get('urlPaths').get('outUrls')
    if out_urls is None or len(out_urls) == 0:
        return resp
    new_resp = resp
    if isinstance(resp, str):
        new_resp = json.loads(resp)
    for out_url in out_urls:
        output_key = out_url['outputKey']
        file_type = out_url['fileType']
        new_resp.get(output_key)
        if file_type == 'png' or file_type == 'jpg':
            content = new_resp.get(output_key)
            _, img_encode = cv2.imencode('.' + file_type, content)
            img_bytes = img_encode.tobytes()
            return type(img_bytes)
        else:
            out_mem_file = io.BytesIO()
            out_mem_file.write(new_resp.get(output_key))
            return type(out_mem_file)

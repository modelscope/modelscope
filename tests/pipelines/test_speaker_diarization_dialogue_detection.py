# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from typing import Any, Dict

import numpy as np

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class SpeakerDiarizationDialogueDetectionTest(unittest.TestCase):

    test_datasets = [{
        'sentence':
        '还有什么双面胶啥的不都是她写的吗？然后这部剧她为了写这部剧，她还说亲自去武汉。然后体验了一下穿防护服的感受。据说那个防护服要穿好几层那种。',
        'label': False
    }, {
        'sentence':
        '你们那儿小区不能用健康宝吗？不能。北一区都可以了。外面进去的就像是那个快递员儿呀，或者是外卖小哥呀，要健康宝。然后本小区的要出入证，都问有出入证吗？',
        'label': True
    }, {
        'sentence': '侦探小说从19世纪中期开始发展。美国作家埃德加‧爱伦‧坡被认为是西方侦探小说的鼻祖。',
        'label': False
    }]

    dialogue_detection_model_id = 'damo/speech_bert_dialogue-detetction_speaker-diarization_chinese'

    def setUp(self) -> None:
        self.task = Tasks.speaker_diarization_dialogue_detection

    def run_pipeline(self,
                     model_id: str,
                     model_revision=None) -> Dict[str, Any]:
        dialogue_detection = pipeline(
            task=self.task, model=model_id, model_revision=model_revision)
        outputs_list = []
        for test_item in self.test_datasets:
            sentence = test_item['sentence']
            outputs_list.append((sentence, dialogue_detection(sentence)))
        return outputs_list

    @unittest.skipUnless(test_level() >= 0, 'Skip test in current test level')
    def test_dialogue_detection_model(self):
        logger.info('Run speaker diarization dialogue detection from modelhub')
        output_list = self.run_pipeline(
            model_id=self.dialogue_detection_model_id, model_revision='v0.5.3')
        for sentence, result in output_list:
            label = result['labels'][np.argmax(result['scores'])]
            logger.info(f'Sentence = {sentence}, label = {label}')


if __name__ == '__main__':
    unittest.main()

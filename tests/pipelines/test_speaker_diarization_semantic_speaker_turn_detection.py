# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from typing import Any, Dict, Optional

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class SpeakerDiarizationSemanticSpeakerTurnDetectionTest(unittest.TestCase):

    test_datasets = [{
        'sentence': '嗯，到时候有问题我再跟您联系吧，刘老师。行，可以的，那到时候再联系吧。',
    }, {
        'sentence':
        '你是如何看待这个问题的呢？这个问题挺好解决的，我们只需要增加停车位就行了。嗯嗯，好，那我们业主就放心了。'
    }, {
        'sentence': '这个电台播放各种音乐，包括古典、爵士、民族等各种风格，并附有专业的音乐解说。'
    }]

    semantic_std_model_id = 'damo/speech_bert_semantic-spk-turn-detection-punc_speaker-diarization_chinese'

    def setUp(self) -> None:
        self.task = Tasks.speaker_diarization_semantic_speaker_turn_detection

    def run_pipeline(self,
                     model_id: str,
                     model_revision=None) -> Dict[str, Any]:
        speaker_turn_detection = pipeline(
            task=self.task, model=model_id, model_revision=model_revision)
        output_list = []
        for sentence_item in self.test_datasets:
            sentence = sentence_item['sentence']
            output_list.append((sentence, speaker_turn_detection(sentence)))
        return output_list

    @unittest.skipUnless(test_level() >= 0, 'Skip test in current test level')
    def test_semantic_speaker_turn_detection_model(self):
        logger.info('Run speaker diarization speaker turn detection')

        pipeline_results = self.run_pipeline(
            model_id=self.semantic_std_model_id, model_revision='v0.5.0')
        for sentence, result in pipeline_results:
            cur_predict_sentence = ''
            predict = result['prediction']
            for i, ch in enumerate(sentence):
                cur_predict_sentence += ch
                if i >= len(predict):
                    continue
                if predict[i] == 1:
                    cur_predict_sentence += '|'
            logger.info(f'\nresult = {result.keys()}'
                        f'\nsentence = {sentence}'
                        f'\npredict  = {cur_predict_sentence}')
        logger.info('Text semantic_speaker_turn_detection model finished')


if __name__ == '__main__':
    unittest.main()

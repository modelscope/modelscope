# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

# import modelscope
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class ProSTTextVideoRetrievalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.task = Tasks.text_video_retrieval
        self.model_id = 'damo/multi_modal_clip_vtretrieval_prost'

    video_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/multi_modal_test_video_9770.mp4'
    caption = 'a person is connecting something to system'
    _input = {'video': video_path, 'text': caption}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        pipeline_prost_text_video_retrieval = pipeline(
            Tasks.text_video_retrieval, model=self.model_id)
        output = pipeline_prost_text_video_retrieval(self._input)
        logger.info('t2v sim: {}'.format(output['textvideo_sim']))
        logger.info('phrase prototype: {}'.format(
            output['phrase_prototype'].shape))
        logger.info('object prototype: {}'.format(
            output['object_prototype'].shape))
        logger.info('sentence prototype: {}'.format(
            output['sentence_prototype'].shape))
        logger.info('event prototype: {}'.format(
            output['event_prototype'].shape))


if __name__ == '__main__':
    unittest.main()

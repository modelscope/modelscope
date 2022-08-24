# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

logger = get_logger()


class VideoMultiModalEmbeddingTest(unittest.TestCase):

    model_id = 'damo/multi_modal_clip_vtretrival_msrvtt_53'
    video_path = 'data/test/videos/multi_modal_test_video_9770.mp4'
    caption = ('a person is connecting something to system', None, None)
    _input = {'video': video_path, 'text': caption}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        pipeline_video_multi_modal_embedding = pipeline(
            Tasks.video_multi_modal_embedding, model=self.model_id)
        output = pipeline_video_multi_modal_embedding(self._input)
        logger.info('text feature: {}'.format(
            output['text_embedding'][0][0][0]))
        logger.info('video feature: {}'.format(
            output['video_embedding'][0][0][0]))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_default_model(self):
        pipeline_video_multi_modal_embedding = pipeline(
            task=Tasks.video_multi_modal_embedding)
        output = pipeline_video_multi_modal_embedding(self._input)
        logger.info('text feature: {}'.format(
            output['text_embedding'][0][0][0]))
        logger.info('video feature: {}'.format(
            output['video_embedding'][0][0][0]))


if __name__ == '__main__':
    unittest.main()

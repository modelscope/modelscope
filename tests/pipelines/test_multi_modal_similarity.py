# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class MultiModalSimilarityTest(unittest.TestCase):
    model_id = 'damo/multi-modal_team-vit-large-patch14_multi-modal-similarity'
    test_img = 'data/test/images/multimodal_similarity.jpg'
    test_str1 = '一个上了年纪的女人在城镇中骑着自行车一个黄色出租车正要从她身边驶过'
    test_str2 = '穿着蓝色连衣裙的那个女人正冲着行来的车辆伸出她的手'

    def infer_pipeline(self, multi_modal_similarity_pipeline):
        test_input1 = {'img': self.test_img, 'text': self.test_str1}
        test_input2 = {'img': self.test_img, 'text': self.test_str2}
        output1 = multi_modal_similarity_pipeline(test_input1)
        output2 = multi_modal_similarity_pipeline(test_input2)
        print('image: {}, text: {}, similarity: {}'.format(
            self.test_img, self.test_str1, output1['scores']))
        print('image: {}, text: {}, similarity: {}'.format(
            self.test_img, self.test_str2, output2['scores']))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        multi_modal_similarity_pipeline = pipeline(
            Tasks.multi_modal_similarity, model=self.model_id)
        self.infer_pipeline(multi_modal_similarity_pipeline)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        multi_modal_similarity_pipeline = pipeline(
            task=Tasks.multi_modal_similarity)
        self.infer_pipeline(multi_modal_similarity_pipeline)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        model = Model.from_pretrained(self.model_id)
        multi_modal_similarity_pipeline = pipeline(
            task=Tasks.multi_modal_similarity, model=model)
        self.infer_pipeline(multi_modal_similarity_pipeline)


if __name__ == '__main__':
    unittest.main()

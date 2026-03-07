# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.models import Model
from modelscope.models.nlp.deberta_v2 import (DebertaV2ForMaskedLM,
                                              DebertaV2Model)
from modelscope.utils.constant import Tasks
from modelscope.utils.import_utils import exists


class DebertaV2BackboneTest(unittest.TestCase):

    @unittest.skipUnless(
        exists('transformers<5.0'),
        'Skip because transformers version is too high.')
    def test_load_model(self):
        model = Model.from_pretrained(
            'damo/nlp_debertav2_fill-mask_chinese-lite')
        self.assertTrue(model.__class__ == DebertaV2ForMaskedLM)
        model = Model.from_pretrained(
            'damo/nlp_debertav2_fill-mask_chinese-lite', task=Tasks.backbone)
        self.assertTrue(model.__class__ == DebertaV2Model)


if __name__ == '__main__':
    unittest.main()

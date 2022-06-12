# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from modelscope.trainers import build_trainer


class DummyTrainerTest(unittest.TestCase):

    def test_dummy(self):
        default_args = dict(cfg_file='configs/examples/train.json')
        trainer = build_trainer('dummy', default_args)

        trainer.train()
        trainer.evaluate()


if __name__ == '__main__':
    unittest.main()

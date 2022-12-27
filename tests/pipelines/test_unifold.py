# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class UnifoldProteinStructureTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.protein_structure
        self.model_id = 'DPTech/uni-fold-monomer'
        self.model_id_multimer = 'DPTech/uni-fold-multimer'

        self.protein = 'MGLPKKALKESQLQFLTAGTAVSDSSHQTYKVSFIENGVIKNAFYKKLDPKNHYPELLAKISVAVSLFKRIFQGRRSAEERLVFDD'
        self.protein_multimer = 'GAMGLPEEPSSPQESTLKALSLYEAHLSSYIMYLQTFLVKTKQKVNNKNYPEFTLFDTSKLKKDQTLKSIKT' + \
            'NIAALKNHIDKIKPIAMQIYKKYSKNIP NIAALKNHIDKIKPIAMQIYKKYSKNIP'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        model_dir1 = snapshot_download(self.model_id_multimer)
        multi_pipeline_ins = pipeline(task=self.task, model=model_dir1)
        _ = multi_pipeline_ins(self.protein_multimer)

        model_dir = snapshot_download(self.model_id)
        mono_pipeline_ins = pipeline(task=self.task, model=model_dir)
        _ = mono_pipeline_ins(self.protein)


if __name__ == '__main__':
    unittest.main()

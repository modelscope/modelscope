# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.audio import AudioBrainPreprocessor
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level

MIX_SPEECH_FILE = 'data/test/audios/mix_speech.wav'
S1_SPEECH_FILE = 'data/test/audios/s1_speech.wav'
S2_SPEECH_FILE = 'data/test/audios/s2_speech.wav'


class TestSeparationTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/speech_mossformer_separation_temporal_8k'

        csv_path = os.path.join(self.tmp_dir, 'test.csv')
        mix_path = os.path.join(os.getcwd(), MIX_SPEECH_FILE)
        s1_path = os.path.join(os.getcwd(), S1_SPEECH_FILE)
        s2_path = os.path.join(os.getcwd(), S2_SPEECH_FILE)
        with open(csv_path, 'w') as w:
            w.write(f'id,mix_wav:FILE,s1_wav:FILE,s2_wav:FILE\n'
                    f'0,{mix_path},{s1_path},{s2_path}\n')
        self.dataset = MsDataset.load(
            'csv', data_files={
                'test': [csv_path]
            }).to_torch_dataset(
                preprocessors=[
                    AudioBrainPreprocessor(
                        takes='mix_wav:FILE', provides='mix_sig'),
                    AudioBrainPreprocessor(
                        takes='s1_wav:FILE', provides='s1_sig'),
                    AudioBrainPreprocessor(
                        takes='s2_wav:FILE', provides='s2_sig')
                ],
                to_tensor=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skip
    def test_trainer(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            max_epochs=2,
            work_dir=self.tmp_dir)
        trainer = build_trainer(
            Trainers.speech_separation, default_args=kwargs)
        # model placement
        trainer.model.load_check_point(device=trainer.device)
        trainer.train()

        logging_path = os.path.join(self.tmp_dir, 'train_log.txt')
        self.assertTrue(
            os.path.exists(logging_path),
            f'Cannot find logging file {logging_path}')
        save_dir = os.path.join(self.tmp_dir, 'save')
        checkpoint_dirs = os.listdir(save_dir)
        self.assertEqual(
            len(checkpoint_dirs), 2, f'Cannot find checkpoint in {save_dir}!')

    @unittest.skip
    def test_eval(self):
        kwargs = dict(
            model=self.model_id,
            train_dataset=None,
            eval_dataset=self.dataset,
            max_epochs=2,
            work_dir=self.tmp_dir)
        trainer = build_trainer(
            Trainers.speech_separation, default_args=kwargs)
        result = trainer.evaluate(None)
        self.assertTrue('si-snr' in result)


if __name__ == '__main__':
    unittest.main()

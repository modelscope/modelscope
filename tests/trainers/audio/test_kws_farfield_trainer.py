import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level

POS_FILE = 'data/test/audios/wake_word_with_label_xyxy.wav'
NEG_FILE = 'data/test/audios/speech_with_noise.wav'
NOISE_FILE = 'data/test/audios/speech_with_noise.wav'
INTERF_FILE = 'data/test/audios/speech_with_noise.wav'
REF_FILE = 'data/test/audios/farend_speech.wav'
NOISE_2CH_FILE = 'data/test/audios/noise_2ch.wav'


class TestKwsFarfieldTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        print(f'tmp dir: {self.tmp_dir}')
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya'

        train_pos_list = self.create_list('pos.list', POS_FILE)
        train_neg_list = self.create_list('neg.list', NEG_FILE)
        train_noise1_list = self.create_list('noise.list', NOISE_FILE)
        train_noise2_list = self.create_list('noise_2ch.list', NOISE_2CH_FILE)
        train_interf_list = self.create_list('interf.list', INTERF_FILE)
        train_ref_list = self.create_list('ref.list', REF_FILE)

        base_dict = dict(
            train_pos_list=train_pos_list,
            train_neg_list=train_neg_list,
            train_noise1_list=train_noise1_list)
        fintune_dict = dict(
            train_pos_list=train_pos_list,
            train_neg_list=train_neg_list,
            train_noise1_list=train_noise1_list,
            train_noise2_type='1',
            train_noise1_ratio='0.2',
            train_noise2_list=train_noise2_list,
            train_interf_list=train_interf_list,
            train_ref_list=train_ref_list)
        self.custom_conf = dict(
            basetrain_easy=base_dict,
            basetrain_normal=base_dict,
            basetrain_hard=base_dict,
            finetune_easy=fintune_dict,
            finetune_normal=fintune_dict,
            finetune_hard=fintune_dict)

    def create_list(self, list_name, audio_file):
        pos_list_file = os.path.join(self.tmp_dir, list_name)
        with open(pos_list_file, 'w') as f:
            for i in range(10):
                f.write(f'{os.path.join(os.getcwd(), audio_file)}\n')
        train_pos_list = f'{pos_list_file}, 1.0'
        return train_pos_list

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_normal(self):
        kwargs = dict(
            model=self.model_id,
            work_dir=self.tmp_dir,
            workers=2,
            max_epochs=2,
            train_iters_per_epoch=2,
            val_iters_per_epoch=1,
            custom_conf=self.custom_conf)

        trainer = build_trainer(
            Trainers.speech_dfsmn_kws_char_farfield, default_args=kwargs)
        trainer.train()
        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files,
                      f'work_dir:{self.tmp_dir}')

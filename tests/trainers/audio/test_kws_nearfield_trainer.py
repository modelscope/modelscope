import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.hub import read_config, snapshot_download
from modelscope.utils.test_utils import test_level
from modelscope.utils.torch_utils import get_dist_info

POS_FILE = 'data/test/audios/kws_xiaoyunxiaoyun.wav'
NEG_FILE = 'data/test/audios/kws_bofangyinyue.wav'


class TestKwsNearfieldTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        print(f'tmp dir: {self.tmp_dir}')
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/speech_charctc_kws_phone-xiaoyun'

        model_dir = snapshot_download(self.model_id)
        print(model_dir)
        self.configs = read_config(self.model_id)

        # update some configs
        self.configs.train.max_epochs = 10
        self.configs.train.batch_size_per_gpu = 4
        self.configs.train.dataloader.workers_per_gpu = 1
        self.configs.evaluation.batch_size_per_gpu = 4
        self.configs.evaluation.dataloader.workers_per_gpu = 1

        self.config_file = os.path.join(self.tmp_dir, 'config.json')
        self.configs.dump(self.config_file)

        self.train_scp, self.cv_scp, self.trans_file = self.create_list()

        print(f'test level is {test_level()}')

    def create_list(self):
        train_scp_file = os.path.join(self.tmp_dir, 'train.scp')
        cv_scp_file = os.path.join(self.tmp_dir, 'cv.scp')
        trans_file = os.path.join(self.tmp_dir, 'merged.trans')

        with open(trans_file, 'w') as fp_trans:
            with open(train_scp_file, 'w') as fp_scp:
                for i in range(8):
                    fp_scp.write(
                        f'train_pos_wav_{i}\t{os.path.join(os.getcwd(), POS_FILE)}\n'
                    )
                    fp_trans.write(f'train_pos_wav_{i}\t小云小云\n')

                for i in range(16):
                    fp_scp.write(
                        f'train_neg_wav_{i}\t{os.path.join(os.getcwd(), NEG_FILE)}\n'
                    )
                    fp_trans.write(f'train_neg_wav_{i}\t播放音乐\n')

            with open(cv_scp_file, 'w') as fp_scp:
                for i in range(2):
                    fp_scp.write(
                        f'cv_pos_wav_{i}\t{os.path.join(os.getcwd(), POS_FILE)}\n'
                    )
                    fp_trans.write(f'cv_pos_wav_{i}\t小云小云\n')

                for i in range(2):
                    fp_scp.write(
                        f'cv_neg_wav_{i}\t{os.path.join(os.getcwd(), NEG_FILE)}\n'
                    )
                    fp_trans.write(f'cv_neg_wav_{i}\t播放音乐\n')

        return train_scp_file, cv_scp_file, trans_file

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_normal(self):
        print('test start ...')
        kwargs = dict(
            model=self.model_id,
            work_dir=self.tmp_dir,
            cfg_file=self.config_file)

        trainer = build_trainer(
            Trainers.speech_kws_fsmn_char_ctc_nearfield, default_args=kwargs)

        kwargs = dict(
            train_data=self.train_scp,
            cv_data=self.cv_scp,
            trans_data=self.trans_file)
        trainer.train(**kwargs)

        rank, _ = get_dist_info()
        if rank == 0:
            results_files = os.listdir(self.tmp_dir)
            for i in range(self.configs.train.max_epochs):
                self.assertIn(f'{i}.pt', results_files)

            kwargs = dict(
                test_dir=self.tmp_dir,
                gpu=-1,
                keywords='小云小云',
                batch_size=4,
            )
            trainer.evaluate(None, None, **kwargs)

            results_files = os.listdir(self.tmp_dir)
            self.assertIn('convert.kaldi.txt', results_files)

        print('test finished ...')


if __name__ == '__main__':
    unittest.main()

import unittest
import zipfile
from pathlib import Path

from maas_lib.fileio import File
from maas_lib.trainers import build_trainer
from maas_lib.utils.logger import get_logger

logger = get_logger()


class SequenceClassificationTrainerTest(unittest.TestCase):

    def test_sequence_classification(self):
        model_url = 'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com' \
                    '/release/easynlp_modelzoo/alibaba-pai/bert-base-sst2.zip'
        cache_path_str = r'.cache/easynlp/bert-base-sst2.zip'
        cache_path = Path(cache_path_str)

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch(exist_ok=True)
            with cache_path.open('wb') as ofile:
                ofile.write(File.read(model_url))

        with zipfile.ZipFile(cache_path_str, 'r') as zipf:
            zipf.extractall(cache_path.parent)

        path: str = './configs/nlp/sequence_classification_trainer.yaml'
        default_args = dict(cfg_file=path)
        trainer = build_trainer('bert-sentiment-analysis', default_args)
        trainer.train()
        trainer.evaluate()


if __name__ == '__main__':
    unittest.main()
    ...

import subprocess
import sys
import unittest

from modelscope.models.cv.anydoor.ldm.util import exists
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class SentenceEmbeddingPipelineTest(unittest.TestCase):

    def setUp(self) -> None:

        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'transformers>=4.51.3'])
        self.model_id = 'Qwen/Qwen3-Embedding-0.6B'
        self.queries = [
            'What is the capital of China?',
            'Explain gravity',
        ]
        self.documents = [
            'The capital of China is Beijing.',
            'Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and '
            'is responsible for the movement of planets around the sun.',
        ]

    @unittest.skipUnless(
        exists('sentence_transformers'),
        'Skip because sentence_transformers is not installed.')
    def test_ori_pipeline(self):
        ppl = pipeline(
            Tasks.sentence_embedding,
            model=self.model_id,
            model_revision='master',
        )
        inputs = {'source_sentence': self.documents}
        embeddings = ppl(input=inputs)['text_embedding']
        self.assertEqual(embeddings.shape[0], len(self.documents))
        self.assertLess((embeddings[0][0] + 0.0471825), 0.01)  # check value

    def test_sentence_embedding_input(self):
        ppl = pipeline(
            Tasks.sentence_embedding,
            model=self.model_id,
            model_revision='master',
        )
        embeddings = ppl(self.queries, prompt_name='query')
        self.assertEqual(embeddings.shape[0], len(self.queries))
        self.assertLess((embeddings[0][0] + 0.050865322), 0.01)  # check value


if __name__ == '__main__':
    unittest.main()

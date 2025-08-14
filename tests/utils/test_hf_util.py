# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import time
import unittest
import uuid

import torch
from huggingface_hub import CommitInfo, RepoUrl

from modelscope import HubApi
from modelscope.utils.hf_util.patcher import patch_context
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1, TEST_MODEL_ORG,
                                         test_level)

logger = get_logger()


class HFUtilTest(unittest.TestCase):

    def setUp(self):
        logger.info('SetUp')
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)
        self.user = TEST_MODEL_ORG
        print(self.user)
        self.create_model_name = '%s/%s_%s' % (self.user, 'test_model_upload',
                                               uuid.uuid4().hex)
        logger.info('create %s' % self.create_model_name)
        temporary_dir = tempfile.mkdtemp()
        self.work_dir = temporary_dir
        self.model_dir = os.path.join(temporary_dir, self.create_model_name)
        self.repo_path = os.path.join(self.work_dir, 'repo_path')
        self.test_folder = os.path.join(temporary_dir, 'test_folder')
        self.test_file1 = os.path.join(
            os.path.join(temporary_dir, 'test_folder', '1.json'))
        self.test_file2 = os.path.join(os.path.join(temporary_dir, '2.json'))
        os.makedirs(self.test_folder, exist_ok=True)
        with open(self.test_file1, 'w') as f:
            f.write('{}')
        with open(self.test_file2, 'w') as f:
            f.write('{}')

        self.pipeline_qa_context = r"""
            Extractive Question Answering is the task of extracting an answer from a text given a question. An example
            of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would
            like to fine-tune a model on a SQuAD task, you may leverage the
            examples/pytorch/question-answering/run_squad.py script.
            """
        self.pipeline_qa_question = 'What is a good example of a question answering dataset?'

    def tearDown(self):
        logger.info('TearDown')
        shutil.rmtree(self.model_dir, ignore_errors=True)
        try:
            self.api.delete_model(model_id=self.create_model_name)
        except Exception:
            pass

    def test_auto_tokenizer(self):
        from modelscope import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'baichuan-inc/Baichuan2-7B-Chat',
            trust_remote_code=True,
            revision='v1.0.3')
        self.assertEqual(tokenizer.vocab_size, 125696)
        self.assertEqual(tokenizer.model_max_length, 4096)
        self.assertFalse(tokenizer.is_fast)

    def test_quantization_import(self):
        from modelscope import BitsAndBytesConfig
        self.assertTrue(BitsAndBytesConfig is not None)

    def test_auto_model(self):
        from modelscope import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            'baichuan-inc/baichuan-7B', trust_remote_code=True)
        self.assertTrue(model is not None)

    def test_auto_config(self):
        from modelscope import AutoConfig, GenerationConfig
        config = AutoConfig.from_pretrained(
            'baichuan-inc/Baichuan-13B-Chat',
            trust_remote_code=True,
            revision='v1.0.3')
        self.assertEqual(config.model_type, 'baichuan')
        gen_config = GenerationConfig.from_pretrained(
            'baichuan-inc/Baichuan-13B-Chat',
            trust_remote_code=True,
            revision='v1.0.3')
        self.assertEqual(gen_config.assistant_token_id, 196)

    def test_qwen_tokenizer(self):
        from modelscope import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            'Qwen/Qwen2-Math-7B-Instruct')
        self.assertTrue(tokenizer is not None)

    def test_extra_ignore_args(self):
        from modelscope import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            'Qwen/Qwen2-Math-7B-Instruct', ignore_file_pattern=[r'\w+\.h5'])
        self.assertTrue(tokenizer is not None)

    def test_transformer_patch(self):
        with patch_context():
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertIsNotNone(tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertIsNotNone(model)

    def test_patch_model(self):
        from modelscope.utils.hf_util.patcher import patch_context
        with patch_context():
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertTrue(model is not None)
        try:
            model = AutoModel.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
        except Exception:
            pass
        else:
            self.assertTrue(False)

    def test_patch_config_bert(self):
        from transformers import BertConfig
        try:
            BertConfig.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
        except Exception:
            pass
        else:
            self.assertTrue(False)

    def test_patch_config(self):
        with patch_context():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertTrue(getattr(config, 'base_model_prefix') == 'encoder')
        try:
            config = AutoConfig.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertTrue(
                getattr(config, 'base_model_prefix', None) != 'encoder')
        except:  # noqa
            pass

        # Test patch again
        with patch_context():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                'iic/nlp_structbert_sentiment-classification_chinese-tiny')
            self.assertTrue(config is not None)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_patch_diffusers(self):
        with patch_context():
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                'AI-ModelScope/stable-diffusion-v1-5')
            self.assertTrue(pipe is not None)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                'AI-ModelScope/stable-diffusion-v1-5')
        except Exception:
            pass
        else:
            self.assertTrue(False)

        from modelscope import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            'AI-ModelScope/stable-diffusion-v1-5')
        self.assertTrue(pipe is not None)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_patch_peft(self):
        with patch_context():
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen1.5-0.5B-Chat',
                trust_remote_code=True,
                torch_dtype=torch.float32)
            model = PeftModel.from_pretrained(
                model,
                'tastelikefeet/test_lora',
                trust_remote_code=True,
                torch_dtype=torch.float32)
            self.assertTrue(model is not None)
        self.assertFalse(hasattr(PeftModel, '_from_pretrained_origin'))

    def test_patch_file_exists(self):
        with patch_context():
            from huggingface_hub import file_exists
            self.assertTrue(
                file_exists('AI-ModelScope/stable-diffusion-v1-5',
                            'feature_extractor/preprocessor_config.json'))
        try:
            # Import again
            from huggingface_hub import file_exists  # noqa
            exists = file_exists('AI-ModelScope/stable-diffusion-v1-5',
                                 'feature_extractor/preprocessor_config.json')
        except Exception:
            pass
        else:
            self.assertFalse(exists)

    def test_patch_file_download(self):
        with patch_context():
            from huggingface_hub import hf_hub_download
            local_dir = hf_hub_download(
                'AI-ModelScope/stable-diffusion-v1-5',
                'feature_extractor/preprocessor_config.json')
            logger.info('patch file_download dir: ' + local_dir)
            self.assertTrue(local_dir is not None)

    def test_patch_create_repo(self):
        with patch_context():
            from huggingface_hub import create_repo
            repo_url: RepoUrl = create_repo(self.create_model_name)
            logger.info('patch create repo result: ' + repo_url.repo_id)
            self.assertTrue(repo_url is not None)
            from huggingface_hub import upload_folder
            commit_info: CommitInfo = upload_folder(
                repo_id=self.create_model_name,
                folder_path=self.test_folder,
                path_in_repo='')
            logger.info('patch create repo result: ' + commit_info.commit_url)
            self.assertTrue(commit_info is not None)
            from huggingface_hub import file_exists
            time.sleep(1)
            self.assertTrue(file_exists(self.create_model_name, '1.json'))
            from huggingface_hub import upload_file
            commit_info: CommitInfo = upload_file(
                path_or_fileobj=self.test_file2,
                path_in_repo='test_folder2',
                repo_id=self.create_model_name)
            time.sleep(1)
            self.assertTrue(
                file_exists(self.create_model_name, 'test_folder2/2.json'))

    def test_who_am_i(self):
        with patch_context():
            from huggingface_hub import whoami
            self.assertTrue(whoami()['name'] == self.user)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_push_to_hub(self):
        with patch_context():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen1.5-0.5B-Chat', trust_remote_code=True)
            model.push_to_hub(self.create_model_name)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_pipeline_model_id(self):
        from modelscope import pipeline
        model_id = 'damotestx/distilbert-base-cased-distilled-squad'
        qa = pipeline('question-answering', model=model_id)
        assert qa(
            question=self.pipeline_qa_question,
            context=self.pipeline_qa_context)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_pipeline_auto_model(self):
        from modelscope import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
        model_id = 'damotestx/distilbert-base-cased-distilled-squad'
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        qa = pipeline('question-answering', model=model, tokenizer=tokenizer)
        assert qa(
            question=self.pipeline_qa_question,
            context=self.pipeline_qa_context)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_pipeline_save_pretrained(self):
        from modelscope import pipeline
        model_id = 'damotestx/distilbert-base-cased-distilled-squad'

        pipe_ori = pipeline('question-answering', model=model_id)

        result_ori = pipe_ori(
            question=self.pipeline_qa_question,
            context=self.pipeline_qa_context)

        # save_pretrained
        repo_id = self.create_model_name
        save_dir = './tmp_test_hf_pipeline'
        try:
            os.system(f'rm -rf {save_dir}')
            self.api.delete_model(repo_id)
            # wait for delete repo
            import time
            time.sleep(5)
        except Exception:
            # if repo not exists
            pass
        pipe_ori.save_pretrained(save_dir, push_to_hub=True, repo_id=repo_id)

        # load from saved
        pipe_new = pipeline('question-answering', model=repo_id)
        result_new = pipe_new(
            question=self.pipeline_qa_question,
            context=self.pipeline_qa_context)

        assert result_new == result_ori


if __name__ == '__main__':
    unittest.main()

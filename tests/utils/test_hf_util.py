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
        response, _ = self.api.login(TEST_ACCESS_TOKEN1)
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
        except Exception as e:
            logger.warning(
                f'Failed to delete model {self.create_model_name}: {e}')

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

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
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
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
            self.assertIsNotNone(tokenizer)
            model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
            self.assertIsNotNone(model)

    def test_patch_model(self):
        from modelscope.utils.hf_util.patcher import patch_context
        with patch_context():
            from transformers import AutoModel
            model = AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B')
            self.assertTrue(model is not None)
        try:
            model = AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B')
        except Exception:
            pass
        else:
            self.assertTrue(False)

    def test_patch_config(self):
        with patch_context():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')
            self.assertTrue(config is not None)
        try:
            AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')
            self.assertTrue(False)
        except:  # noqa
            pass

        # Test patch again
        with patch_context():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')
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

    def test_automapping_download(self):
        from modelscope import AutoConfig
        model = 'nomic-ai/nomic-embed-text-v1.5'
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        model_dir = config.name_or_path
        files = os.listdir(model_dir)
        has_weight_files = any(
            f.endswith('.safetensors') or f.endswith('.bin') for f in files)
        self.assertFalse(
            has_weight_files,
            f'Expected no weight files in {model_dir}, but found: '
            f"{[f for f in files if f.endswith('.safetensors') or f.endswith('.bin')]}"
        )
        # modelscope_hub 0.1.x layout uses models/{owner}--{name}/...;
        # older layout used {owner}/{name}/.  Accept either.
        cache_root = model_dir
        for _ in range(4):
            parent = os.path.dirname(cache_root)
            if parent == cache_root:
                break
            cache_root = parent
            candidates = [
                os.path.join(cache_root, 'nomic-ai', 'nomic-bert-2048'),
                os.path.join(cache_root, 'models',
                             'nomic-ai--nomic-bert-2048'),
            ]
            for model_dir_2 in candidates:
                if not os.path.exists(model_dir_2):
                    continue
                # Walk into snapshots/{rev} if present.
                check_dirs = [model_dir_2]
                snapshots = os.path.join(model_dir_2, 'snapshots')
                if os.path.isdir(snapshots):
                    check_dirs.extend(
                        os.path.join(snapshots, d)
                        for d in os.listdir(snapshots)
                        if os.path.isdir(os.path.join(snapshots, d)))
                for check_dir in check_dirs:
                    files = os.listdir(check_dir)
                    has_weight_files = any(
                        f.endswith('.safetensors') or f.endswith('.bin')
                        for f in files)
                    self.assertFalse(
                        has_weight_files,
                        f'Expected no weight files in {check_dir}, but found: '
                        f"{[f for f in files if f.endswith('.safetensors') or f.endswith('.bin')]}"
                    )

    def test_dynamic_module_double_dash_cache_path(self):
        """Cross-repo auto_map must survive cache paths that contain '--'.

        modelscope_hub 0.1.x stores repos under ``models/{owner}--{name}/``.
        Rejoining that path into ``class_reference`` with ``--`` makes
        transformers' ``split("--")`` raise ValueError.
        """
        from unittest import mock

        from modelscope.utils.hf_util.patcher import \
            _get_class_from_dynamic_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        local_path = os.path.join(tmp, 'models', 'nomic-ai--nomic-bert-2048',
                                  'snapshots', 'rev')
        os.makedirs(local_path)
        pretrained = os.path.join(tmp, 'models',
                                  'nomic-ai--nomic-embed-text-v1.5',
                                  'snapshots', 'rev')
        os.makedirs(pretrained)

        captured = {}

        def fake_origin(class_reference, pretrained_model_name_or_path, *args,
                        **kwargs):
            # Signature must match transformers so has_pretrained_arg is True.
            captured['class_reference'] = class_reference
            captured['pretrained'] = pretrained_model_name_or_path
            return type('DummyConfig', (), {})

        class_ref = ('nomic-ai/nomic-bert-2048--'
                     'configuration_hf_nomic_bert.NomicBertConfig')

        # create=True: do not permanently leave origin_* on the module
        # (would break test_import_not_pollute_dynamic_module).
        with mock.patch(
                'transformers.dynamic_module_utils.origin_get_class_from_dynamic_module',
                new=fake_origin,
                create=True):
            with mock.patch(
                    'modelscope.snapshot_download', return_value=local_path):
                _get_class_from_dynamic_module(class_ref, pretrained)

        # Must pass bare module.Class (no '--') so transformers does not split.
        self.assertEqual(captured['class_reference'],
                         'configuration_hf_nomic_bert.NomicBertConfig')
        # Local cache path (which contains '--') is pretrained_model_name_or_path.
        self.assertEqual(captured['pretrained'], local_path)

    def test_dynamic_module_remote_pretrained_tuple_args(self):
        """Remote pretrained_model_name_or_path must not mutate args in place.

        ``*args`` is a tuple; ``args[0] = snapshot_download(...)`` raises
        TypeError.  Rebuild the tuple instead (regression from 9379504f).
        """
        from unittest import mock

        from modelscope.utils.hf_util.patcher import \
            _get_class_from_dynamic_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        downloaded = os.path.join(tmp, 'models', 'org--model', 'snapshots',
                                  'rev')
        os.makedirs(downloaded)

        captured = {}

        def fake_origin(class_reference, pretrained_model_name_or_path, *args,
                        **kwargs):
            captured['class_reference'] = class_reference
            captured['pretrained'] = pretrained_model_name_or_path
            return type('DummyConfig', (), {})

        # No '--' in class_reference: only the pretrained download branch runs.
        remote_id = 'org/model-not-on-disk'
        with mock.patch(
                'transformers.dynamic_module_utils.origin_get_class_from_dynamic_module',
                new=fake_origin,
                create=True):
            with mock.patch(
                    'modelscope.snapshot_download',
                    return_value=downloaded) as sd:
                # Must not raise TypeError: 'tuple' object does not support
                # item assignment.
                _get_class_from_dynamic_module('modeling.Foo', remote_id)

        sd.assert_called_once_with(remote_id, local_files_only=False)
        self.assertEqual(captured['class_reference'], 'modeling.Foo')
        self.assertEqual(captured['pretrained'], downloaded)

    def test_dynamic_module_local_files_only_forwarded(self):
        """Download kwargs must be forwarded to both snapshot_download calls.

        Cross-repo auto_map references previously omitted local_files_only /
        cache_dir / token / code_revision, so offline and custom-cache loads
        still hit the wrong download path for the referenced repo.
        """
        from unittest import mock

        from modelscope.utils.hf_util.patcher import \
            _get_class_from_dynamic_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        downloaded = os.path.join(tmp, 'models', 'org--model', 'snapshots',
                                  'rev')
        cross_repo = os.path.join(tmp, 'models', 'org--other', 'snapshots',
                                  'rev')
        os.makedirs(downloaded)
        os.makedirs(cross_repo)

        def fake_origin(class_reference, pretrained_model_name_or_path, *args,
                        **kwargs):
            return type('DummyConfig', (), {})

        remote_id = 'org/model-not-on-disk'
        class_ref = 'org/other--configuration_foo.FooConfig'
        call_kwargs = []
        cache_dir = os.path.join(tmp, 'custom_cache')
        token = 'ms-test-token'

        def fake_download(repo_id, **kwargs):
            call_kwargs.append((repo_id, dict(kwargs)))
            if repo_id == remote_id:
                return downloaded
            if repo_id == 'org/other':
                return cross_repo
            raise AssertionError(f'unexpected download: {repo_id}')

        with mock.patch(
                'transformers.dynamic_module_utils.origin_get_class_from_dynamic_module',
                new=fake_origin,
                create=True):
            with mock.patch(
                    'modelscope.snapshot_download', side_effect=fake_download):
                _get_class_from_dynamic_module(
                    class_ref,
                    pretrained_model_name_or_path=remote_id,
                    local_files_only=True,
                    cache_dir=cache_dir,
                    token=token,
                    revision='model-rev',
                    code_revision='code-rev')

        self.assertEqual(len(call_kwargs), 2)
        self.assertEqual(call_kwargs[0][0], remote_id)
        self.assertEqual(
            call_kwargs[0][1], {
                'local_files_only': True,
                'cache_dir': cache_dir,
                'token': token,
                'revision': 'model-rev',
            })
        self.assertEqual(call_kwargs[1][0], 'org/other')
        self.assertEqual(call_kwargs[1][1]['local_files_only'], True)
        self.assertEqual(call_kwargs[1][1]['cache_dir'], cache_dir)
        self.assertEqual(call_kwargs[1][1]['token'], token)
        self.assertEqual(call_kwargs[1][1]['revision'], 'code-rev')
        self.assertIn('ignore_file_pattern', call_kwargs[1][1])

    def test_ms_download_kwargs_from_hf(self):
        """Shared HF→MS download kwargs mapping used by patcher download paths."""
        from modelscope.utils.hf_util.patcher import _ms_download_kwargs_from_hf

        self.assertEqual(
            _ms_download_kwargs_from_hf({}), {'local_files_only': False})
        got = _ms_download_kwargs_from_hf(
            {
                'local_files_only': True,
                'cache_dir': '/tmp/c',
                'token': 'sekrit',
                'token_ignored': True,
            },
            revision='main')
        self.assertEqual(
            got, {
                'local_files_only': True,
                'cache_dir': '/tmp/c',
                'token': 'sekrit',
                'revision': 'master',
            })
        # HF token=True means "default creds"; only string tokens are forwarded.
        self.assertNotIn('token', _ms_download_kwargs_from_hf({'token': True}))

    def test_get_model_dir_forwards_cache_dir_and_token(self):
        """from_pretrained download path must forward cache_dir and token."""
        from unittest import mock

        from modelscope import AutoConfig

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(tmp, 'config.json'), 'w') as f:
            f.write('{"model_type": "bert", "hidden_size": 8}')

        cache_dir = os.path.join(tmp, 'custom_cache')
        token = 'ms-test-token'
        with mock.patch(
                'modelscope.snapshot_download', return_value=tmp) as sd:
            AutoConfig.from_pretrained(
                'org/model-not-on-disk',
                cache_dir=cache_dir,
                token=token,
                local_files_only=True)

        sd.assert_called_once()
        _, kwargs = sd.call_args
        self.assertEqual(kwargs.get('cache_dir'), cache_dir)
        self.assertEqual(kwargs.get('token'), token)
        self.assertTrue(kwargs.get('local_files_only'))

    def test_dynamic_module_pretrained_via_kwargs(self):
        """pretrained_model_name_or_path may be passed as a keyword argument."""
        from unittest import mock

        from modelscope.utils.hf_util.patcher import \
            _get_class_from_dynamic_module

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        downloaded = os.path.join(tmp, 'models', 'org--model', 'snapshots',
                                  'rev')
        cross_repo = os.path.join(tmp, 'models', 'org--other', 'snapshots',
                                  'rev')
        os.makedirs(downloaded)
        os.makedirs(cross_repo)

        captured = {}

        def fake_origin(class_reference, pretrained_model_name_or_path, *args,
                        **kwargs):
            captured['class_reference'] = class_reference
            captured['pretrained'] = pretrained_model_name_or_path
            captured['kwargs'] = kwargs
            return type('DummyConfig', (), {})

        remote_id = 'org/model-not-on-disk'
        class_ref = 'org/other--configuration_foo.FooConfig'

        def fake_download(repo_id, **kwargs):
            if repo_id == remote_id:
                return downloaded
            if repo_id == 'org/other':
                return cross_repo
            raise AssertionError(f'unexpected download: {repo_id}')

        with mock.patch(
                'transformers.dynamic_module_utils.origin_get_class_from_dynamic_module',
                new=fake_origin,
                create=True):
            with mock.patch(
                    'modelscope.snapshot_download', side_effect=fake_download):
                # Keyword form: must download and not pass duplicate positional.
                _get_class_from_dynamic_module(
                    class_ref, pretrained_model_name_or_path=remote_id)

        self.assertEqual(captured['class_reference'],
                         'configuration_foo.FooConfig')
        self.assertEqual(captured['pretrained'], cross_repo)
        self.assertNotIn('pretrained_model_name_or_path', captured['kwargs'])

    def test_import_not_pollute_dynamic_module(self):
        """Importing from modelscope must not globally patch
        transformers' get_class_from_dynamic_module (issue #1751).

        The patch should be scoped to each from_pretrained / get_config_dict
        call via _dynamic_module_patch_scope(), leaving the global state clean
        for unrelated ``from transformers import AutoConfig`` callers.
        """
        from modelscope import AutoConfig
        from modelscope.utils.file_utils import get_modelscope_cache_dir
        from transformers import dynamic_module_utils

        # 1. Importing AutoConfig from modelscope (wrap=True) must NOT
        #    globally patch get_class_from_dynamic_module.
        self.assertFalse(
            hasattr(dynamic_module_utils,
                    'origin_get_class_from_dynamic_module'),
            'Importing AutoConfig from modelscope must not globally patch '
            'get_class_from_dynamic_module (issue #1751)')

        # 2. The modelscope AutoConfig should still work with
        #    trust_remote_code, downloading through ModelScope (not HF).
        model = 'nomic-ai/nomic-embed-text-v1.5'
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        model_dir = config.name_or_path

        # Verify files are in the ModelScope cache, not the HF cache.
        ms_cache = get_modelscope_cache_dir()
        self.assertTrue(
            os.path.realpath(model_dir).startswith(os.path.realpath(ms_cache)),
            f'Model files should be in ModelScope cache ({ms_cache}), '
            f'but found at {model_dir}')

        # 3. After the call the global patch should still NOT be in effect
        #    (the _dynamic_module_patch_scope was temporary).
        self.assertFalse(
            hasattr(dynamic_module_utils,
                    'origin_get_class_from_dynamic_module'),
            'Global get_class_from_dynamic_module should not be patched '
            'after a modelscope AutoConfig call (issue #1751)')

        # 4. Pure transformers AutoConfig (without modelscope patching) must
        #    NOT be redirected to ModelScope.  With the old code the global
        #    patch would intercept this call and download from ModelScope;
        #    with the fix the call goes to HuggingFace directly.
        from transformers import AutoConfig as HFAutoConfig
        try:
            hf_config = HFAutoConfig.from_pretrained(
                model, trust_remote_code=True)
            hf_model_dir = hf_config.name_or_path
            # If the download succeeded, files must be outside the
            # ModelScope cache (i.e. in the HuggingFace cache).
            self.assertFalse(
                os.path.realpath(hf_model_dir).startswith(
                    os.path.realpath(ms_cache)),
                f'Transformers AutoConfig should download from HuggingFace, '
                f'not ModelScope (issue #1751). Found files at: '
                f'{hf_model_dir}')
        except Exception:
            # If HuggingFace is unreachable the call fails — which is also
            # correct: it means the request did NOT go through ModelScope
            # (which would have succeeded).
            pass

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
            try:
                self.api.delete_model(repo_id)
            except Exception as e:
                logger.warning(f'Failed to delete model {repo_id}: {e}')
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

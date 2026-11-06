# Copyright (c) Alibaba, Inc. and its affiliates.
"""datasets compatibility tests with real builders and isolated transports."""
import ast
import copy
import hashlib
import importlib
import inspect
import io
import os
import pickle
import tempfile
import textwrap
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

import datasets
import httpx
import json
import pyarrow as pa
import requests
from datasets import (Audio, ClassLabel, Dataset, Features, Image,
                      IterableDataset, LargeList, List, Sequence, Value)
from datasets.features import features as hf_features
from datasets.utils import file_utils
from huggingface_hub import HfApi, HfFileSystem

from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.utils import hf_datasets_util
from modelscope.msdatasets.utils._module_factories import \
    _compat_local_script_module
from modelscope.utils.constant import Hubs

_SCRIPT = textwrap.dedent('''\
    import json
    import os
    from pathlib import Path

    import datasets
    from .helper import LABEL

    marker = os.environ.get('MODELSCOPE_TEST_SCRIPT_MARKER')
    if marker:
        Path(marker).write_text('executed', encoding='utf-8')

    class TinyConfig(datasets.BuilderConfig):
        def __init__(self, custom_label='default:', **kwargs):
            super().__init__(**kwargs)
            self.custom_label = custom_label

    class Tiny(datasets.GeneratorBasedBuilder):
        BUILDER_CONFIG_CLASS = TinyConfig
        BUILDER_CONFIGS = [TinyConfig(name='small'), TinyConfig(name='other')]
        DEFAULT_CONFIG_NAME = 'small'

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({
                'id': datasets.Value('int32'),
                'text': datasets.Value('string'),
            }))

        def _split_generators(self, dl_manager):
            paths = dl_manager.download_and_extract(self.config.data_files or {
                'train': 'data/train-0.jsonl', 'test': 'data/test.jsonl',
            })
            return [datasets.SplitGenerator(name=name, gen_kwargs={'files': files})
                    for name, files in paths.items()]

        def _generate_examples(self, files):
            if isinstance(files, str):
                files = [files]
            i = 0
            for file in files:
                with open(file, encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        row['text'] = LABEL + self.config.custom_label + row['text']
                        yield i, row
                        i += 1
''')


class DatasetsCompatTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'tiny'
        self.root.mkdir()
        self.cache = self.root / 'cache'
        self.data_dir = self.root / 'data'
        self.data_dir.mkdir()
        self.rows = [{'id': i, 'text': f'row-{i}'} for i in range(12)]
        for shard in range(3):
            self._write_jsonl(f'train-{shard}.jsonl',
                              self.rows[shard * 4:(shard + 1) * 4])
        self._write_jsonl('test.jsonl', self.rows[:2])
        (self.root / 'tiny.py').write_text(_SCRIPT, encoding='utf-8')
        (self.root / 'helper.py').write_text(
            "LABEL = 'helper:'\n", encoding='utf-8')
        self.features = Features({
            'id': Value('int32'),
            'text': Value('string')
        })
        # Cache dynamic modules locally without touching existing user scripts.
        patcher = mock.patch.object(datasets.config, 'HF_MODULES_CACHE',
                                    str(self.root / 'modules'))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_jsonl(self, name, rows):
        path = self.data_dir / name
        path.write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return str(path)

    def _load(self, path='json', **kwargs):
        kwargs.setdefault('cache_dir', str(self.cache))
        kwargs.setdefault('custom_cfg', None)
        return MsDataset.load(path, **kwargs)

    def _script(self, **kwargs):
        kwargs.setdefault('trust_remote_code', True)
        kwargs.setdefault('split', 'train')
        return self._load(str(self.root / 'tiny.py'), **kwargs)

    def test_all_datasets_imports_in_source_tree(self):
        source_root = Path(__file__).resolve().parents[2] / 'modelscope'
        checked = set()
        for source in source_root.rglob('*.py'):
            tree = ast.parse(
                source.read_text(encoding='utf-8'), filename=str(source))
            guarded = {
                id(child)
                for node in ast.walk(tree) if isinstance(node, ast.Try)
                for child in ast.walk(node)
            }
            for node in ast.walk(tree):
                if (not isinstance(node, ast.ImportFrom) or node.level
                        or not node.module
                        or not (node.module == 'datasets'
                                or node.module.startswith('datasets.'))
                        or id(node) in guarded):
                    continue
                for alias in node.names:
                    if alias.name == '*' or (node.module,
                                             alias.name) in checked:
                        continue
                    with self.subTest(
                            source=str(source.relative_to(source_root)),
                            symbol=alias.name):
                        module = importlib.import_module(node.module)
                        if not hasattr(module, alias.name):
                            importlib.import_module(
                                f'{node.module}.{alias.name}')
                    checked.add((node.module, alias.name))
        self.assertGreater(len(checked), 30)

    def test_public_load_signature(self):
        signature = inspect.signature(MsDataset.load)
        self.assertIs(signature.parameters['trust_remote_code'].default, False)
        self.assertIs(signature.parameters['use_streaming'].default, False)
        self.assertIn('config_kwargs', signature.parameters)

    def test_local_json_data_and_return_types(self):
        ds = self._load(
            data_files={'train': str(self.data_dir / 'train-*.jsonl')},
            split='train')
        self.assertIsInstance(ds, MsDataset)
        self.assertEqual(list(ds), self.rows)
        both = self._load(
            data_files={
                'train': str(self.data_dir / 'train-0.jsonl'),
                'test': str(self.data_dir / 'test.jsonl')
            })
        self.assertEqual(set(both), {'train', 'test'})
        self.assertTrue(
            all(isinstance(value, MsDataset) for value in both.values()))

    def test_local_explicit_features(self):
        ds = self._load(
            data_files=str(self.data_dir / 'train-0.jsonl'),
            split='train',
            features=self.features)
        self.assertEqual(ds.to_hf_dataset().features, self.features)

    def test_local_csv_and_parquet(self):
        csv_path = self.root / 'tiny.csv'
        csv_path.write_text('id,text\n1,one\n2,two\n', encoding='utf-8')
        parquet_path = self.root / 'tiny.parquet'
        pa.parquet.write_table(pa.Table.from_pylist(self.rows), parquet_path)
        for path, expected in [(csv_path, [{
                'id': 1,
                'text': 'one'
        }, {
                'id': 2,
                'text': 'two'
        }]), (parquet_path, self.rows)]:
            with self.subTest(format=path.suffix):
                self.assertEqual(list(self._load(str(path))), expected)

    def test_split_slicing_and_combination(self):
        files = {
            'train': str(self.data_dir / 'train-*.jsonl'),
            'test': str(self.data_dir / 'test.jsonl')
        }
        self.assertEqual(
            list(self._load(data_files=files, split='train[:2]+test')),
            self.rows[:2] * 2)
        with self.assertRaises(ValueError):
            self._load(data_files=files, split='missing')

    def test_local_script_real_builder_and_custom_kwargs(self):
        ds = self._script(custom_label='custom:')
        self.assertEqual(ds.to_hf_dataset().features, self.features)
        self.assertEqual(ds[0], {'id': 0, 'text': 'helper:custom:row-0'})

    def test_local_script_default_config_and_directory(self):
        ds = self._load(str(self.root), trust_remote_code=True, split='train')
        self.assertEqual(ds[0]['text'], 'helper:default:row-0')

    def test_local_script_untrusted_never_executes(self):
        marker = self.root / 'executed'
        with mock.patch.dict(os.environ,
                             {'MODELSCOPE_TEST_SCRIPT_MARKER': str(marker)}):
            for trust in (False, None):
                with self.subTest(trust=trust), self.assertRaisesRegex(
                        ValueError, 'trust_remote_code=True'):
                    self._script(trust_remote_code=trust)
                self.assertFalse(marker.exists())

    def test_cached_script_does_not_bypass_trust(self):
        self._script()
        for trust in (False, None):
            with self.subTest(trust=trust), self.assertRaisesRegex(
                    ValueError, 'trust_remote_code=True'):
                _compat_local_script_module(
                    str(self.root / 'tiny.py'), trust_remote_code=trust)

    def test_script_info_only_uses_compat_factory(self):
        result = hf_datasets_util.DatasetsWrapperHF.load_dataset(
            str(self.root / 'tiny.py'),
            trust_remote_code=True,
            cache_dir=str(self.cache),
            dataset_info_only=True)
        self.assertEqual(result, {
            'small': ['test', 'train'],
            'other': ['test', 'train']
        })

    def _info_only_without_downloads(self, path='json', **kwargs):
        with ExitStack() as stack:
            for target in (
                    'requests.sessions.Session.send',
                    'httpx.Client._send_single_request',
                    'datasets.download.download_manager.DownloadManager.download',
                    'datasets.download.download_manager.DownloadManager.extract'
            ):
                stack.enter_context(
                    mock.patch(
                        target,
                        side_effect=AssertionError(
                            'unexpected data download')))
            result = self._load(path, dataset_info_only=True, **kwargs)
        self.assertFalse(list(self.cache.rglob('*.arrow')))
        return result

    def test_packaged_info_only_uses_effective_data_files(self):
        csv_path = self.data_dir / 'records.csv'
        csv_path.write_text('id,text\n1,one\n', encoding='utf-8')
        for builder, path in [('json', self.data_dir / 'train-0.jsonl'),
                              ('csv', csv_path)]:
            for files, splits in [(str(path), ['train']),
                                  ([str(path)], ['train']),
                                  ({
                                      'validation': str(path),
                                      'test': [str(path)]
                                  }, ['validation', 'test'])]:
                for name in (None, 'selected'):
                    with self.subTest(builder=builder, files=files, name=name):
                        self.assertEqual(
                            self._info_only_without_downloads(
                                builder, data_files=files, subset_name=name),
                            {name or 'default': splits})

    def test_local_info_only_infers_file_and_directory_splits(self):
        self.assertEqual(
            self._info_only_without_downloads(
                str(self.data_dir / 'train-0.jsonl')), {'default': ['train']})
        for path, kwargs in [(str(self.data_dir), {}),
                             ('json', {
                                 'data_dir': str(self.data_dir)
                             })]:
            with self.subTest(path=path):
                self.assertEqual(
                    self._info_only_without_downloads(path, **kwargs),
                    {'default': ['train', 'test']})

    def test_info_only_preserves_other_metadata_configs(self):
        readme = _README.replace('data/', '')
        (self.data_dir / 'README.md').write_text(readme, encoding='utf-8')
        expected = {'small': ['train', 'test'], 'other': ['train']}
        overridden = {'small': ['validation'], 'other': ['train']}
        self.assertEqual(
            self._info_only_without_downloads(str(self.data_dir)), expected)
        self.assertEqual(
            self._info_only_without_downloads(
                str(self.data_dir),
                subset_name='small',
                data_files={'validation': str(self.data_dir / 'test.jsonl')}),
            overridden)
        self.assertEqual(
            self._info_only_without_downloads(
                str(self.data_dir), subset_name='other'), expected)

    def test_script_info_only_preserves_generated_split_names(self):
        script = _SCRIPT.replace('name=name, gen_kwargs=',
                                 "name='validation', gen_kwargs=")
        (self.root / 'tiny.py').write_text(script, encoding='utf-8')
        expected = {'small': ['validation'], 'other': ['validation']}
        for path in (str(self.root / 'tiny.py'), str(self.root)):
            with self.subTest(path=path):
                self.assertEqual(
                    self._info_only_without_downloads(
                        path,
                        data_files={
                            'source': str(self.data_dir / 'test.jsonl')
                        },
                        trust_remote_code=True), expected)

    def test_info_only_uses_metadata_without_split_discovery(self):
        script = _SCRIPT.replace(
            'return datasets.DatasetInfo(features=',
            'return datasets.DatasetInfo(splits=datasets.SplitDict({'
            '"train": datasets.SplitInfo(name="train", num_examples=12, '
            'num_bytes=0)}), features=')
        script = script.replace(
            '        paths = dl_manager.download_and_extract(',
            "        raise AssertionError('unexpected split discovery')\n"
            '        paths = dl_manager.download_and_extract(')
        (self.root / 'tiny.py').write_text(script, encoding='utf-8')
        expected = {'small': ['train'], 'other': ['train']}
        self.assertEqual(
            self._info_only_without_downloads(
                str(self.root), trust_remote_code=True), expected)

    def test_info_only_discovery_errors_are_not_empty_success(self):
        script = _SCRIPT.replace(
            '        paths = dl_manager.download_and_extract(',
            "        raise RuntimeError('split discovery failed') "
            "from ValueError('fixture cause')\n"
            '        paths = dl_manager.download_and_extract(')
        (self.root / 'tiny.py').write_text(script, encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,
                                    'split discovery failed') as caught:
            self._info_only_without_downloads(
                str(self.root), trust_remote_code=True)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(str(caught.exception.__cause__), 'fixture cause')
        builder = hf_datasets_util.DatasetsWrapperHF.load_dataset_builder(
            str(self.root), trust_remote_code=True, cache_dir=str(self.cache))
        self.assertEqual(
            hf_datasets_util._discover_splits_from_builder(builder), set())

    def test_dry_run_download_config_and_nested_paths_are_isolated(self):
        options = {'fixture': {'values': ['original']}}
        original_options = copy.deepcopy(options)
        config = datasets.DownloadConfig(
            token=False,
            storage_options=copy.deepcopy(options),
            extract_on_the_fly=False)
        original = copy.deepcopy(config)
        manager = hf_datasets_util._DryRunDownloadManager(config)
        other = hf_datasets_util._DryRunDownloadManager(config)
        manager.download_config.storage_options['fixture']['values'].append(
            'changed')
        manager.download_config.extract_on_the_fly = True
        self.assertEqual(config, original)
        self.assertEqual(other.download_config, original)
        files = {'train': ['first', 'second'], 'test': ('third', )}
        self.assertEqual(
            manager.download_and_extract(files), {
                'train': [os.devnull, os.devnull],
                'test': (os.devnull, )
            })
        builder = hf_datasets_util.DatasetsWrapperHF.load_dataset_builder(
            'json',
            data_files={
                'train': str(self.data_dir / 'train-*.jsonl'),
                'test': str(self.data_dir / 'test.jsonl')
            },
            cache_dir=str(self.cache),
            features=self.features,
            token=False,
            storage_options=options)
        # Native builders may add defaults; preserve the full initialized state.
        builder_options = copy.deepcopy(builder.storage_options)
        self.assertEqual(
            hf_datasets_util._discover_splits_from_builder(
                builder, download_config=config, strict=True),
            {'train', 'test'})
        self.assertEqual(config, original)
        self.assertEqual(builder.storage_options, builder_options)
        self.assertEqual(builder.storage_options['fixture'],
                         original_options['fixture'])
        self.assertEqual(options, original_options)

    def test_info_only_missing_data_files_still_fails(self):
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            self._info_only_without_downloads(data_dir=str(empty))
        with self.assertRaises(FileNotFoundError):
            self._info_only_without_downloads(
                data_files=str(self.data_dir / 'missing.jsonl'))

    def test_streaming_split_limits_and_skip_take(self):
        files = {
            'train': str(self.data_dir / 'train-*.jsonl'),
            'test': str(self.data_dir / 'test.jsonl')
        }
        for split in ('train[:2]', 'train[2:5]', 'train[:25%]',
                      'train[:2]+test[:2]'):
            with self.subTest(split=split), self.assertRaisesRegex(
                    ValueError, 'Bad split'):
                self._load(data_files=files, split=split, use_streaming=True)
        plain = self._load(
            data_files=files, split='train',
            use_streaming=True).to_hf_dataset()
        self.assertEqual(list(plain.skip(2).take(3)), self.rows[2:5])
        self.assertEqual(
            list(
                self._load(
                    data_files=files, split='train+test', use_streaming=True)),
            self.rows + self.rows[:2])

    def test_local_streaming_after_load_and_repeat(self):
        ds = self._load(
            data_files={'train': str(self.data_dir / 'train-*.jsonl')},
            split='train',
            use_streaming=True)
        self.assertIsInstance(ds.to_hf_dataset(), IterableDataset)
        self.assertEqual(list(ds), self.rows)
        self.assertEqual(list(ds), self.rows)

    def test_script_streaming_after_load(self):
        ds = self._script(use_streaming=True)
        self.assertEqual([row['id'] for row in ds], list(range(4)))
        self.assertEqual(
            ds[0]['text']
            if not isinstance(ds.to_hf_dataset(), IterableDataset) else next(
                iter(ds))['text'], 'helper:default:row-0')

    def test_features_round_trip_and_sequence_semantics(self):
        feature_set = Features({
            'sequence':
            Sequence({
                'x': Value('int32'),
                'y': Value('string')
            }),
            'list':
            List({'x': Value('int32')}),
            'large':
            LargeList(Value('string')),
            'label':
            ClassLabel(names=['no', 'yes']),
            'image':
            Image(decode=False),
            'audio':
            Audio(decode=False),
            'array':
            datasets.Array2D(shape=(2, 2), dtype='float32'),
        })
        legacy = feature_set.to_dict()
        legacy['sequence'] = {
            '_type': 'Sequence',
            'feature': {
                'x': {
                    '_type': 'Value',
                    'dtype': 'int32'
                },
                'y': {
                    '_type': 'Value',
                    'dtype': 'string'
                },
            }
        }
        actual = Features(hf_datasets_util.generate_from_dict_ms(legacy))
        self.assertEqual(actual, feature_set)
        self.assertEqual(actual.arrow_schema, feature_set.arrow_schema)
        self.assertEqual(
            Features.from_dict(feature_set.to_dict()), feature_set)

    def test_json_null_and_nullable_numpy(self):
        values = {'payload': [None, {'x': 1}], 'id': [None, 2]}
        ds = Dataset.from_dict(
            values,
            features=Features({
                'payload': datasets.Json(),
                'id': Value('int64')
            }))
        self.assertIsNone(ds[0]['payload'])
        self.assertEqual(ds[1]['payload'], {'x': 1})
        self.assertEqual(ds.features['id'], Value('int64'))
        path = self.root / 'nullable.jsonl'
        ds.to_json(str(path))
        exported = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertIsNone(exported[0]['id'])
        self.assertEqual(exported[1]['id'], 2)

    def test_map_filter_split_concatenate_and_round_trip(self):
        ds = Dataset.from_list(self.rows, features=self.features)
        mapped = ds.map(lambda row: {'id': row['id'] + 1})
        self.assertEqual(len(mapped.filter(lambda row: row['id'] > 6)), 6)
        parts = ds.train_test_split(test_size=0.25, seed=42)
        combined = datasets.concatenate_datasets(
            [parts['train'], parts['test']])
        self.assertEqual(sorted(combined['id']), list(range(12)))
        self.assertEqual(
            MsDataset.to_ms_dataset(ds).to_hf_dataset().features,
            self.features)
        saved = self.root / 'saved'
        ds.save_to_disk(str(saved))
        self.assertEqual(list(datasets.load_from_disk(str(saved))), self.rows)

    def test_shuffle_and_checkpoint_resume_twice(self):
        ds = self._load(
            data_files={
                'train': str(self.data_dir / 'train-*.jsonl')
            },
            split='train',
            use_streaming=True).to_hf_dataset()
        for shards in (1, 10):
            with self.subTest(max_buffer_input_shards=shards):
                options = {'seed': 42, 'buffer_size': 4}
                if 'max_buffer_input_shards' in inspect.signature(
                        ds.shuffle).parameters:
                    options['max_buffer_input_shards'] = shards
                shuffled = ds.shuffle(**options)
                expected = list(shuffled)
                self.assertEqual(
                    sorted(row['id'] for row in expected), list(range(12)))
                self.assertEqual(list(ds.shuffle(**options)), expected)
                iterator = iter(shuffled)
                prefix = [next(iterator) for _ in range(3)]
                state = copy.deepcopy(shuffled.state_dict())
                restored = ds.shuffle(**options)
                restored.load_state_dict(state)
                remaining = list(restored)
                # Assert exact restoration separately without a shuffle buffer.
                self.assertTrue(
                    all(row in self.rows for row in prefix + remaining))
        plain = ds
        iterator = iter(plain)
        prefix = [next(iterator) for _ in range(3)]
        state = copy.deepcopy(plain.state_dict())
        restored = self._load(
            data_files={
                'train': str(self.data_dir / 'train-*.jsonl')
            },
            split='train',
            use_streaming=True).to_hf_dataset()
        restored.load_state_dict(state)
        self.assertEqual(prefix + list(restored), self.rows)
        restored.load_state_dict(state)
        self.assertEqual(prefix + list(restored), self.rows)

    def test_streaming_map_filter_and_batch(self):
        ds = self._load(
            data_files={
                'train': str(self.data_dir / 'train-*.jsonl')
            },
            split='train',
            use_streaming=True).to_hf_dataset()
        changed = ds.map(lambda row: {
            'id': row['id'] + 1
        }).filter(lambda row: row['id'] % 2 == 0)
        self.assertEqual([row['id'] for row in changed], [2, 4, 6, 8, 10, 12])
        self.assertEqual([len(batch['id']) for batch in ds.iter(batch_size=5)],
                         [5, 5, 2])

    def test_download_config_copy_is_isolated(self):
        from modelscope.msdatasets.download.download_config import DataDownloadConfig

        config = DataDownloadConfig(
            storage_options={'modelscope': {
                'token': 'first-fixture'
            }},
            oss_config={'key': ['original']})
        copied = config.copy()
        copied.storage_options['modelscope']['token'] = 'second-fixture'
        copied.oss_config['key'].append('changed')
        self.assertEqual(config.storage_options['modelscope']['token'],
                         'first-fixture')
        self.assertEqual(config.oss_config, {'key': ['original']})
        self.assertEqual(DataDownloadConfig().oss_config, {})

    def test_native_registered_features_are_preserved(self):
        with mock.patch.dict(hf_features._FEATURE_TYPES,
                             {'CompatValue': Value}):
            self.assertEqual(
                hf_datasets_util.generate_from_dict_ms(
                    {'tag': {
                        '_type': 'CompatValue',
                        'dtype': 'string'
                    }}), {'tag': Value('string')})

    def test_training_args_dataset_file(self):
        from modelscope.trainers.training_args import build_dataset_from_file

        path = self.root / 'training.json'
        path.write_text(
            json.dumps([{
                'dataset': {
                    'dataset_name': 'json',
                    'data_files': str(self.data_dir / 'train-*.jsonl'),
                    'split': 'train',
                    'cache_dir': str(self.cache),
                    'custom_cfg': None,
                    'features': None
                },
                'column_mapping': {
                    'id': 'label',
                    'text': 'sentence'
                },
                'usage': 0.75,
            }]),
            encoding='utf-8')
        train, validation = build_dataset_from_file(str(path))
        self.assertEqual((len(train), len(validation)), (9, 3))
        self.assertEqual(set(train.column_names), {'sentence', 'label'})
        self.assertEqual(
            sorted(
                datasets.concatenate_datasets([train, validation])['label']),
            list(range(12)))

    def test_torch_streaming_workers(self):
        import torch

        ds = self._load(
            data_files={
                'train': str(self.data_dir / 'train-*.jsonl')
            },
            split='train',
            use_streaming=True).to_hf_dataset()
        loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
        ids = [value for batch in loader for value in batch['id'].tolist()]
        self.assertEqual(sorted(ids), list(range(12)))

    def test_local_cache_reuse_without_network(self):
        options = {
            'data_files': str(self.data_dir / 'train-0.jsonl'),
            'split': 'train'
        }
        first = list(self._load(**options))
        with mock.patch(
                'requests.sessions.Session.send',
                side_effect=AssertionError('unexpected network')):
            self.assertEqual(list(self._load(**options)), first)

    def test_load_context_has_no_process_global_side_effects(self):
        targets = [(datasets.config, 'HF_ENDPOINT'),
                   (file_utils, 'get_from_cache'), (HfApi, 'dataset_info'),
                   (HfApi, 'list_repo_tree'), (HfApi, 'get_paths_info'),
                   (HfFileSystem, '__init__'), (HfFileSystem, '_open'),
                   (hf_features, 'generate_from_dict'),
                   (datasets.data_files, 'resolve_pattern')]
        originals = [(target, name, getattr(target, name))
                     for target, name in targets]
        try:
            for streaming in (False, True):
                with hf_datasets_util.load_dataset_with_ctx(
                        'json',
                        data_files=str(self.data_dir / 'train-0.jsonl'),
                        split='train',
                        cache_dir=str(self.cache),
                        streaming=streaming) as ds:
                    self.assertTrue(next(iter(ds)))
                for target, name, original in originals:
                    self.assertIs(
                        getattr(target, name), original, f'{target}.{name}')
        finally:
            for target, name, original in originals:
                setattr(target, name, original)

    def test_concurrent_native_loads(self):

        def read(shard):
            return list(
                self._load(
                    data_files=str(self.data_dir / f'train-{shard}.jsonl'),
                    split='train',
                    use_streaming=True))

        with ThreadPoolExecutor(max_workers=3) as pool:
            actual = list(pool.map(read, range(3)))
        self.assertEqual([row for shard in actual for row in shard], self.rows)

    def test_torch_adapter_without_models(self):
        import torch
        ds = self._load(
            data_files=str(self.data_dir / 'train-0.jsonl'), split='train')
        torch_ds = ds.to_torch_dataset(columns=['id'], to_tensor=True)
        batch = next(iter(torch.utils.data.DataLoader(torch_ds, batch_size=2)))
        self.assertEqual(batch['id'].tolist(), [0, 1])


_README = '''---
configs:
- config_name: small
  default: true
  data_files:
  - split: train
    path: data/train-*.jsonl
  - split: test
    path: data/test.jsonl
- config_name: other
  data_files:
  - split: train
    path: data/test.jsonl
---
Tiny offline dataset.
'''


class ModelScopeDatasetsCompatTest(unittest.TestCase):
    """Mock HTTP transport only; exercise the real Hub and data loading chain."""

    _write_jsonl = DatasetsCompatTest._write_jsonl
    _load = DatasetsCompatTest._load

    def setUp(self):
        from modelscope_hub.config import HubConfig

        DatasetsCompatTest.setUp(self)
        self.endpoint = 'https://modelscope.test'
        self.requests = []
        self.files = {'README.md': _README.encode()}
        self.files.update({
            f'data/{path.name}': path.read_bytes()
            for path in self.data_dir.iterdir()
        })
        self.metadata_status = 200
        self.tree_status = 200
        self.formation = 4
        self.zero_size = False
        isolated_config = HubConfig(
            endpoint=self.endpoint,
            token='',
            config_dir=self.root / 'home',
            cache_dir=self.root / 'hub-cache')
        for patcher in [
                mock.patch('modelscope_hub.config._default_config',
                           isolated_config),
                mock.patch(
                    'modelscope_hub.compat.hub_api.LegacyHubApi._dataset_id_type_cache',
                    {}),
                mock.patch('huggingface_hub.constants.HF_HUB_CACHE',
                           str(self.root / 'hf-cache')),
                mock.patch(
                    'requests.sessions.Session.send',
                    side_effect=self._send_requests),
                mock.patch(
                    'httpx.Client._send_single_request',
                    side_effect=self._send_httpx)
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(
            hf_datasets_util._ModelScopeFileSystem.clear_instance_cache)

    def _tree(self):
        directories = {
            str(parent)
            for name in self.files for parent in Path(name).parents
            if str(parent) != '.'
        }
        entries = [{
            'type': 'directory',
            'path': name,
            'size': 0,
            'oid': '0' * 40
        } for name in sorted(directories)]
        for name, data in self.files.items():
            entries.append({
                'type': 'file',
                'path': name,
                'size': 0 if self.zero_size else len(data),
                'oid': hashlib.sha256(data).hexdigest()
            })
        return entries

    def _reply(self, request):
        url = urlparse(str(request.url))
        path = url.path
        query = parse_qs(url.query)
        headers = {
            'Content-Type': 'application/json',
            'X-Repo-Commit': '1' * 40
        }
        self.requests.append(
            (request.method, str(request.url), dict(request.headers)))
        if request.method not in ('GET', 'HEAD', 'POST'):
            raise AssertionError(f'Unexpected HTTP method: {request.method}')
        if path.startswith('/openapi/v1/datasets/test/tiny'):
            status = self.metadata_status
            body = {
                'id': 1,
                'owner': 'test',
                'name': 'tiny',
                'visibility': 'public',
                'private': False
            }
        elif path.endswith('/ststoken'):
            status = 200
            body = {
                'AccessId': 'fixture-id',
                'AccessSecret': 'fixture-secret',
                'SecurityToken': 'fixture-sts',
                'Bucket': 'fixture-bucket',
                'Region': 'oss-cn-hangzhou',
                'Dir': 'fixture-data',
                'BackupDir': 'fixture-backup'
            }
        elif path.startswith(
                '/api/v1/datasets/test/tiny') and '/repo' not in path:
            status = 200
            body = {
                'Id': 1,
                'Type': self.formation,
                'Name': 'tiny',
                'Visibility': 5
            }
        elif path.endswith('/repo/tree'):
            status = self.tree_status
            root = query.get('Root', ['/'])[0].strip('/')
            recursive = query.get('Recursive', ['True'])[0].lower() == 'true'
            selected = []
            for item in self._tree():
                if root and not item['path'].startswith(root + '/'):
                    continue
                relative = item['path'][len(root)
                                        + 1:] if root else item['path']
                if recursive or '/' not in relative:
                    selected.append(item)
            size = int(query.get('PageSize', [max(len(selected), 1)])[0])
            number = int(query.get('PageNumber', [1])[0])
            selected = selected[(number - 1) * size:number * size]
            body = {
                'Files': [{
                    'Path':
                    item['path'],
                    'Size':
                    item['size'],
                    'Sha256':
                    item['oid'],
                    'Type':
                    'tree' if item['type'] == 'directory' else 'blob'
                } for item in selected]
            }
        elif path.startswith('/api/datasets/test/tiny'):
            status = 200
            if '/tree/' in path or '/paths-info/' in path:
                body = self._tree()
            else:
                body = {
                    'id': 'test/tiny',
                    'sha': '1' * 40,
                    'private': False,
                    'siblings': [{
                        'rfilename': name
                    } for name in self.files],
                    'cardData': {}
                }
            content = json.dumps(body).encode()
            headers['Content-Length'] = str(len(content))
            return status, headers, content
        elif '/resolve/' in path or path.endswith('/repo'):
            name = (
                query.get('FilePath', [''])[0] if path.endswith('/repo') else
                path.split('/resolve/', 1)[1].split('/', 1)[1])
            data = self.files.get(name)
            if data is None:
                return 404, headers, b'{"Code":404,"Message":"not found"}'
            headers.update({
                'Content-Length': str(len(data)),
                'ETag': '"' + hashlib.sha256(data).hexdigest() + '"',
                'Accept-Ranges': 'bytes'
            })
            requested_range = request.headers.get('Range')
            status = 200
            if requested_range:
                start, end = requested_range.removeprefix('bytes=').split('-')
                start, end = int(start), int(end) if end else len(data) - 1
                headers[
                    'Content-Range'] = f'bytes {start}-{min(end, len(data) - 1)}/{len(data)}'
                data = data[start:end + 1]
                headers['Content-Length'] = str(len(data))
                status = 206
            return status, headers, data if request.method != 'HEAD' else b''
        else:
            raise AssertionError(
                f'Unexpected URL: {request.method} {request.url}')
        envelope = {
            'Code': status,
            'Success': status == 200,
            'Data': body,
            'Message': 'fixture failure' if status != 200 else 'OK'
        }
        content = json.dumps(envelope).encode()
        headers['Content-Length'] = str(len(content))
        return status, headers, content

    def _send_requests(self, request, **kwargs):
        status, headers, content = self._reply(request)
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response.request = request
        response.headers.update(headers)
        response._content = content
        response._content_consumed = True
        response.raw = io.BytesIO(content)
        return response

    def _send_httpx(self, request):
        status, headers, content = self._reply(request)
        return httpx.Response(
            status, headers=headers, content=content, request=request)

    def _remote(self, **kwargs):
        kwargs.setdefault('subset_name', 'small')
        kwargs.setdefault('split', 'train')
        return self._load('test/tiny', **kwargs)

    def _enable_script(self):
        self.files['tiny.py'] = _SCRIPT.encode()
        self.files['helper.py'] = b"LABEL = 'helper:'\n"

    def test_file_listing_is_scoped_and_paginated(self):
        self.files.update({f'unused/{i}.jsonl': b'{}\n' for i in range(110)})
        api = hf_datasets_util._ModelScopeHfApi(endpoint=self.endpoint)
        info = api.dataset_info('test/tiny', revision='master')
        self.assertFalse(
            any(
                item.rfilename.startswith('unused/')
                for item in info.siblings))
        listing = list(
            api.list_repo_tree(
                'test/tiny', path_in_repo='unused', recursive=False))
        self.assertEqual(len(listing), 110)
        pages = [
            parse_qs(urlparse(url).query).get('PageNumber')
            for _, url, _ in self.requests if 'Root=unused' in url
        ]
        self.assertEqual(pages, [['1'], ['2']])

    def test_explicit_data_files_do_not_scan_unrequested_tree(self):
        self.files['unused/secret.jsonl'] = b'{}\n'
        ds = self._remote(
            use_streaming=True, data_files={'train': 'data/train-0.jsonl'})
        self.assertEqual(list(ds), self.rows[:4])
        queries = [
            parse_qs(urlparse(url).query) for _, url, _ in self.requests
            if '/repo/tree' in url
        ]
        self.assertTrue(queries)
        self.assertTrue(
            all(query.get('Recursive') == ['False'] for query in queries))
        self.assertFalse(
            any(query.get('Root') == ['unused'] for query in queries))

    def test_general_dataset_real_load_with_features(self):
        ds = self._remote(features=self.features, num_proc=1)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.features, self.features)
        self.assertEqual(list(ds), self.rows)
        self.assertTrue(
            any('/openapi/v1/' in url for _, url, _ in self.requests))
        self.assertTrue(
            any('/repo/tree' in url for _, url, _ in self.requests))

    def test_general_info_and_custom_subset(self):
        self.assertEqual(
            self._remote(dataset_info_only=True, split=None), {
                'small': ['train', 'test'],
                'other': ['train']
            })
        self.assertEqual(
            list(self._remote(subset_name='other')), self.rows[:2])

    def test_general_split_slice_and_combination(self):
        self.assertEqual(
            list(self._remote(split='train[:2]+test')), self.rows[:2] * 2)
        with self.assertRaisesRegex(ValueError, 'Available splits'):
            self._remote(split='missing')

    def test_remote_stream_after_context_with_native_hf_load(self):
        ms_ds = self._remote(use_streaming=True, token='ms-fixture')
        hf_ds = self._load(
            'test/tiny',
            hub=Hubs.huggingface,
            token='hf-fixture',
            split='train',
            use_streaming=True,
            data_files={'train': 'data/train-0.jsonl'})
        self.assertEqual(list(ms_ds), self.rows)
        self.assertEqual(list(hf_ds), self.rows[:4])
        hf_requests = [
            headers for _, url, headers in self.requests
            if urlparse(url).hostname == 'huggingface.co'
        ]
        self.assertTrue(hf_requests)
        self.assertTrue(
            all('ms-fixture' not in str(headers) for headers in hf_requests))
        self.assertEqual(datasets.config.HF_ENDPOINT, 'https://huggingface.co')
        self.assertIs(HfApi.dataset_info, hf_datasets_util.HfApi.dataset_info)

    def test_remote_script_and_cached_trust_gate(self):
        self._enable_script()
        ds = self._remote(trust_remote_code=True, custom_label='custom:')
        self.assertEqual(ds[0], {'id': 0, 'text': 'helper:custom:row-0'})
        for trust in (False, None):
            with self.subTest(trust=trust), self.assertRaisesRegex(
                    ValueError, 'trust_remote_code=True'):
                self._remote(trust_remote_code=trust)

    def test_remote_script_streaming(self):
        self._enable_script()
        ds = self._remote(trust_remote_code=True, use_streaming=True)
        self.assertEqual([row['id'] for row in ds], list(range(4)))
        self.assertEqual(next(iter(ds))['text'], 'helper:default:row-0')

    def test_concurrent_streaming_and_token_isolation(self):

        def read(token):
            return list(self._remote(use_streaming=True, token=token))

        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(read, ['first-fixture', 'second-fixture']))
        self.assertEqual(actual, [self.rows, self.rows])
        for token in ('first-fixture', 'second-fixture'):
            fs = hf_datasets_util._ModelScopeFileSystem(
                endpoint=self.endpoint, token=token, skip_instance_cache=True)
            self.assertIn(token, fs._api._build_hf_headers()['Cookie'])
        from modelscope_hub.config import get_default_config
        self.assertEqual(get_default_config().token, '')

    def test_legacy_hf_script_chain(self):
        self.formation = 1
        self._enable_script()
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                ds = self._remote(
                    trust_remote_code=True,
                    token='legacy-fixture',
                    custom_label='legacy:',
                    use_streaming=streaming)
                self.assertIsInstance(ds, MsDataset)
                self.assertEqual(
                    next(iter(ds)), {
                        'id': 0,
                        'text': 'helper:legacy:row-0'
                    })
        self.assertFalse(
            any('/ststoken' in url for _, url, _ in self.requests))

    def _enable_native_meta(self):
        self.formation = 2
        self.files['tiny.json'] = json.dumps({
            'small': {
                'train': {
                    'meta': 'data/train.csv'
                },
                'test': {
                    'meta': 'data/test.csv'
                },
            }
        }).encode()
        self.files['data/train.csv'] = (
            'id,text\n' + ''.join(f'{row["id"]},{row["text"]}\n'
                                  for row in self.rows)).encode()
        self.files['data/test.csv'] = b'id,text\n0,row-0\n1,row-1\n'

    def test_native_meta_csv_chain_and_features(self):
        self._enable_native_meta()
        ds = self._remote(token='meta-fixture', features=self.features)
        self.assertIsInstance(ds, MsDataset)
        self.assertEqual(ds.to_hf_dataset().features, self.features)
        self.assertEqual(list(ds), self.rows)
        self.assertFalse(
            any('/ststoken' in url for _, url, _ in self.requests))

    def test_native_meta_streaming_and_multiple_splits(self):
        self._enable_native_meta()
        ds = self._remote(
            token='meta-fixture',
            split=None,
            use_streaming=True,
            features=self.features)
        train = list(ds['train'])
        test = list(ds['test'])
        self.assertEqual([row['id'][0] for row in train], list(range(12)))
        self.assertEqual([row['id'][0] for row in test], [0, 1])
        self.assertEqual(ds['train'].to_hf_dataset().features, self.features)
        self.assertEqual(ds['train'].to_hf_dataset().num_shards, 1)

    def test_native_stream_file_download_uses_oss_client(self):
        self.formation = 2
        self.files['tiny.json'] = json.dumps({
            'small': {
                'train': {
                    'meta': 'data/assets.csv',
                    'file': 'assets',
                    'args': {
                        'big_data': True
                    },
                }
            }
        }).encode()
        self.files['data/assets.csv'] = b'id,asset:FILE\n0,assets/sample.txt\n'

        def download(bucket, key, filename, **kwargs):
            self.assertEqual(key, 'fixture-data/assets/sample.txt')
            Path(filename).write_bytes(b'fixture payload')

        with mock.patch('modelscope.msdatasets.utils.oss_utils.MS_CACHE_HOME', str(self.root)), \
                mock.patch('oss2.Bucket.object_exists', return_value=True), \
                mock.patch('oss2.resumable_download', side_effect=download) as transport:
            ds = self._remote(token='oss-fixture', use_streaming=True)
            row = next(iter(ds))
        self.assertEqual(
            Path(row['asset:FILE'][0]).read_bytes(), b'fixture payload')
        transport.assert_called_once()
        headers = [
            headers for _, url, headers in self.requests if '/ststoken' in url
        ]
        self.assertTrue(headers)
        self.assertTrue(
            all('oss-fixture' in str(header) for header in headers))

    def test_native_info_only_without_data_downloads(self):
        self._enable_native_meta()
        self.assertEqual(
            self._remote(dataset_info_only=True, split=None),
            {'small': ['test', 'train']})
        self.assertFalse(
            any('FilePath=data%2F' in url for _, url, _ in self.requests))

    def test_modelscope_repo_scope_rejects_foreign_files(self):
        with self.assertRaisesRegex(ValueError,
                                    'requested ModelScope dataset'):
            self._remote(
                use_streaming=True,
                data_files={
                    'train':
                    'modelscope://datasets/foreign/repo@master/secret.jsonl'
                })
        self.assertFalse(
            any('foreign/repo' in url for _, url, _ in self.requests))

    def test_private_imagefolder_decoding_after_context_and_pickle(self):
        from PIL import Image as PILImage

        buffer = io.BytesIO()
        PILImage.new('RGB', (3, 2), (200, 10, 20)).save(buffer, format='PNG')
        self.files['images/one.png'] = buffer.getvalue()
        self.files['README.md'] = b'''---
configs:
- config_name: small
  data_files:
  - split: train
    path: images/*.png
---
'''
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                ds = self._remote(
                    use_streaming=streaming, token='image-fixture')
                restored = pickle.loads(pickle.dumps(ds))
                image = next(iter(restored))['image']
                self.assertEqual(image.size, (3, 2))
                self.assertEqual(image.getpixel((0, 0)), (200, 10, 20))
                self.assertNotIn('image-fixture',
                                 json.dumps(ds.features.to_dict()))
        reads = [
            (url, headers) for method, url, headers in self.requests
            if method == 'GET' and ('one.png' in url or 'one%2Epng' in url)
        ]
        self.assertTrue(reads)
        authenticated_reads = all('image-fixture' in str(headers)
                                  for _, headers in reads)
        self.assertTrue(authenticated_reads, reads)

    def test_decoder_keeps_native_lazy_file_handle(self):
        path = 'modelscope://datasets/test/tiny@master/data/train-0.jsonl'
        images = {'image': [{'path': path, 'bytes': None}]}
        ds = Dataset.from_dict(images, features=Features({'image': Image()}))
        hf_datasets_util._bind_modelscope_decoders(
            ds, {
                'modelscope': {
                    'endpoint': self.endpoint,
                    'token': 'lazy-fixture',
                    'repo_id': 'test/tiny',
                }
            })

        def native_decode(feature, value, token_per_repo_id=None):
            self.assertIsNone(value['bytes'])
            return file_utils.xopen(value['path'], 'rb', block_size=0)

        with mock.patch.object(Image, 'decode_example', new=native_decode):
            handle = ds[0]['image']
        self.assertIsNone(hf_datasets_util._decoder_storage_options.get())
        self.assertFalse(
            any(method == 'GET' and 'FilePath=' in url
                for method, url, _ in self.requests))
        with handle:
            self.assertEqual(
                handle.read(1), self.files['data/train-0.jsonl'][:1])
        reads = [
            headers for method, url, headers in self.requests
            if method == 'GET' and 'FilePath=' in url
        ]
        self.assertTrue(
            all('lazy-fixture' in str(headers) for headers in reads))

    def test_remote_parquet_zero_size_and_ranges(self):
        buffer = io.BytesIO()
        pa.parquet.write_table(pa.Table.from_pylist(self.rows), buffer)
        self.files['data/tiny.parquet'] = buffer.getvalue()
        self.zero_size = True
        ds = self._remote(
            use_streaming=True, data_files={'train': 'data/tiny.parquet'})
        self.assertEqual(list(ds), self.rows)

    def test_remote_zip_protocol_chain(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('records.jsonl', self.files['data/train-0.jsonl'])
        self.files['data/tiny.zip'] = buffer.getvalue()
        ds = self._remote(
            use_streaming=True, data_files={'train': 'data/tiny.zip'})
        self.assertEqual(list(ds), self.rows[:4])

    def test_streaming_pickle_and_multishard_shuffle(self):
        ds = self._remote(use_streaming=True, token='pickle-fixture')
        restored = pickle.loads(pickle.dumps(ds))
        self.assertEqual(list(restored), self.rows)
        shuffled = restored.shuffle(seed=42, buffer_size=4)
        self.assertEqual(
            sorted(row['id'] for row in shuffled), list(range(12)))

    def test_filesystem_cache_does_not_reuse_other_default_token(self):
        from modelscope_hub.config import get_default_config

        config = get_default_config()
        config.token = 'first-fixture'
        first = hf_datasets_util._ModelScopeFileSystem(endpoint=self.endpoint)
        config.token = 'second-fixture'
        second = hf_datasets_util._ModelScopeFileSystem(endpoint=self.endpoint)
        self.assertIsNot(first, second)
        self.assertIn('first-fixture',
                      first._api._build_hf_headers()['Cookie'])
        self.assertIn('second-fixture',
                      second._api._build_hf_headers()['Cookie'])

    def test_stream_redirect_drops_modelscope_cookie(self):

        def send(request):
            self.requests.append(
                (request.method, str(request.url), dict(request.headers)))
            if urlparse(str(request.url)).hostname == 'modelscope.test':
                return httpx.Response(
                    302,
                    headers={'Location': 'https://external.test/data'},
                    request=request)
            self.assertNotIn('cookie', request.headers)
            self.assertNotIn('authorization', request.headers)
            return httpx.Response(
                200, content=self.files['data/train-0.jsonl'], request=request)

        ds = self._remote(
            token='redirect-fixture',
            use_streaming=True,
            data_files={'train': 'data/train-0.jsonl'})
        with mock.patch('httpx.Client._send_single_request', side_effect=send):
            self.assertEqual(list(ds), self.rows[:4])

    def test_request_redirect_drops_modelscope_cookie(self):
        for authorization in (None, 'Bearer auth-fixture'):
            with self.subTest(authorization=authorization):
                headers = {'Cookie': 'm_session_id=ms-fixture'}
                if authorization is not None:
                    headers['Authorization'] = authorization
                request = requests.Request(
                    'GET', self.endpoint + '/data', headers=headers).prepare()
                response = requests.Response()
                response.status_code = 302
                response.url = request.url
                response.request = request
                response.headers['Location'] = 'https://external.test/data'
                response._content = b''
                response._content_consumed = True
                with requests.Session() as session:
                    session.trust_env = False
                    redirected = next(
                        session.resolve_redirects(
                            response, request, yield_requests=True))
                self.assertEqual(redirected.url, 'https://external.test/data')
                self.assertNotIn('Cookie', redirected.headers)
                self.assertNotIn('Authorization', redirected.headers)

    def test_metadata_errors_are_structured(self):
        from modelscope.hub.errors import AuthenticationError, PermissionDeniedError

        for status, error_type in [(401, AuthenticationError),
                                   (403, PermissionDeniedError)]:
            self.metadata_status = status
            with self.subTest(
                    status=status), self.assertRaises(error_type) as result:
                self._remote(use_streaming=True)
            self.assertEqual(result.exception.status_code, status)

    def test_tree_errors_are_not_swallowed(self):
        from modelscope.hub.errors import PermissionDeniedError

        self.tree_status = 403
        with self.assertRaises(PermissionDeniedError):
            self._remote(use_streaming=True)

    def test_zero_size_metadata_stream_is_readable(self):
        self.zero_size = True
        ds = self._remote(use_streaming=True)
        self.assertEqual(list(ds), self.rows)

    def test_get_file_reuses_hub_download_and_offline_cache(self):
        fs = hf_datasets_util._ModelScopeFileSystem(
            endpoint=self.endpoint,
            token='ms-fixture',
            cache_dir=str(self.cache),
            skip_instance_cache=True)
        target = io.BytesIO()
        fs.get_file(
            'datasets/test/tiny@master/data/train-0.jsonl', outfile=target)
        self.assertEqual(target.getvalue(), self.files['data/train-0.jsonl'])
        fs = hf_datasets_util._ModelScopeFileSystem(
            endpoint=self.endpoint,
            token='ms-fixture',
            cache_dir=str(self.cache),
            local_files_only=True,
            skip_instance_cache=True)
        with mock.patch(
                'requests.sessions.Session.send',
                side_effect=AssertionError('unexpected network')):
            second = io.BytesIO()
            fs.get_file(
                'datasets/test/tiny@master/data/train-0.jsonl', outfile=second)
        self.assertEqual(second.getvalue(), target.getvalue())


if __name__ == '__main__':
    unittest.main()

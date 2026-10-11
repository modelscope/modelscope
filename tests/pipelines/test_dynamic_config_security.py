# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest
from contextlib import ExitStack
from unittest import mock

from modelscope.pipelines import builder as pipeline_builder
from modelscope.pipelines.audio.linear_aec_pipeline import (
    ArchitectureConfigError, _validate_aec_config, initialize_config)
from modelscope.utils import automodel_utils
from modelscope.utils.architecture import \
    ArchitectureConfigError as SharedArchitectureConfigError
from modelscope.utils.architecture import (instantiate_registered_architecture,
                                           require_trust_remote_code)
from modelscope.utils.config import Config


class ArchitectureRegistrySecurityTest(unittest.TestCase):

    def test_unknown_target_is_rejected_without_loading_a_factory(self):
        factory = mock.Mock()
        with self.assertRaises(SharedArchitectureConfigError):
            instantiate_registered_architecture(
                {
                    'target': 'attacker.payload',
                    'params': {}
                }, {'approved': factory},
                context='test architecture')
        factory.assert_not_called()

    def test_object_schema_rejects_legacy_dynamic_import_fields(self):
        with self.assertRaises(SharedArchitectureConfigError):
            instantiate_registered_architecture(
                {
                    'target': 'approved',
                    'params': {},
                    'module': 'attacker.payload',
                }, {'approved': lambda: object},
                context='test architecture')

    def test_remote_code_gate_is_not_an_assertion(self):
        with self.assertRaises(RuntimeError):
            require_trust_remote_code(False, 'test pipeline')


class PipelineTrustPropagationTest(unittest.TestCase):

    def test_hub_verified_source_propagates_trust_and_skips_remote_repo_code(
            self):
        configuration = Config({
            'pipeline': {
                'type': 'image-view-transform',
            },
            'allow_remote': True,
        })
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    pipeline_builder,
                    'is_model_from_trusted_source',
                    return_value=True))
            stack.enter_context(
                mock.patch.object(
                    pipeline_builder,
                    'is_official_hub_path',
                    return_value=True))
            stack.enter_context(
                mock.patch.object(
                    pipeline_builder,
                    'read_config',
                    return_value=configuration))
            stack.enter_context(
                mock.patch.object(
                    pipeline_builder,
                    'normalize_model_input',
                    return_value='/verified/snapshot'))
            register_plugins = stack.enter_context(
                mock.patch.object(pipeline_builder, 'register_plugins_repo'))
            register_repo = stack.enter_context(
                mock.patch.object(pipeline_builder, 'register_modelhub_repo'))
            build = stack.enter_context(
                mock.patch.object(
                    pipeline_builder, 'build_pipeline', return_value=object()))
            pipeline_builder.pipeline(
                task='image-view-transform', model='damo/known-model')

        register_plugins.assert_called_once_with(None)
        register_repo.assert_called_once_with('/verified/snapshot', False)
        self.assertTrue(
            build.call_args.kwargs['default_args']['trust_remote_code'])
        self.assertTrue(build.call_args.args[0]['trust_remote_code'])


class AecConfigurationSecurityTest(unittest.TestCase):

    @staticmethod
    def _config(module='modelscope.models.audio.aec.network.se_net'):
        return {
            'io': {
                'fbank_config': {
                    'dither': 1.0,
                    'frame_length': 40,
                    'frame_shift': 20,
                    'num_mel_bins': 80,
                    'sample_frequency': 16000,
                    'window_type': 'hamming',
                },
            },
            'nnet': {
                'module': module,
                'main': 'MaskNet',
                'args': {
                    'indim': 240,
                    'outdim': 321,
                    'layers': 12,
                    'hidden_dim': 512,
                    'vad': True,
                },
            },
            'loss': {
                'module': 'network.loss',
                'main': 'mask_loss_function',
                'args': {
                    'loss_func': 'psm_vad_loss_dlen',
                    'n_fft': 640,
                    'hop_length': 320,
                },
            },
        }

    def test_known_legacy_aec_config_maps_to_static_architecture(self):
        config = _validate_aec_config(self._config())
        self.assertEqual(config['nnet']['architecture'], 'mask_net')
        with mock.patch(
                'modelscope.pipelines.audio.linear_aec_pipeline.MaskNet'
        ) as net:
            initialize_config(config['nnet'])
        net.assert_called_once_with(**config['nnet']['args'])

    def test_aec_config_rejects_unapproved_module(self):
        with self.assertRaises(ArchitectureConfigError):
            _validate_aec_config(self._config(module='model.test'))


class TrustedSourceSecurityTest(unittest.TestCase):

    def tearDown(self):
        automodel_utils._is_model_from_trusted_source.cache_clear()

    def test_hub_metadata_not_cache_path_authorizes_source(self):
        automodel_utils._is_model_from_trusted_source.cache_clear()
        with mock.patch(
                'modelscope.hub.api.HubApi.get_model',
                return_value={
                    'Owner': 'iic',
                    'Name': 'known-model',
                }) as get_model:
            self.assertTrue(
                automodel_utils.is_model_from_trusted_source(
                    'damo/known-model'))
        get_model.assert_called_once_with('damo/known-model', revision=None)
        self.assertFalse(
            automodel_utils.is_model_from_trusted_source(
                '/cache/iic--known-model/snapshots/main'))

    def test_untrusted_hub_owner_is_rejected(self):
        automodel_utils._is_model_from_trusted_source.cache_clear()
        with mock.patch(
                'modelscope.hub.api.HubApi.get_model',
                return_value={
                    'Owner': 'attacker',
                    'Name': 'known-model',
                }):
            self.assertFalse(
                automodel_utils.is_model_from_trusted_source(
                    'damo/known-model'))


class SinkWiringSecurityTest(unittest.TestCase):

    def _source(self, relative_path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(
                os.path.join(root, relative_path), encoding='utf-8') as source:
            return source.read()

    def test_gates_precede_configuration_and_construction_sinks(self):
        cases = (
            ('modelscope/pipelines/audio/linear_aec_pipeline.py',
             ('super().__init__', 'yaml.safe_load(')),
            ('modelscope/models/cv/image_view_transform/image_view_transform_infer.py',
             ('super().__init__', '_load_model_config(')),
            ('modelscope/pipelines/cv/image_to_3d_pipeline.py',
             ('super().__init__', '_load_model_config(')),
            ('modelscope/models/cv/anydoor/anydoor_model.py',
             ('super().__init__', )),
        )
        for relative_path, sinks in cases:
            source = self._source(relative_path)
            gate = source.find('require_trust_remote_code(')
            self.assertGreater(gate, -1, relative_path)
            for sink in sinks:
                self.assertGreater(
                    source.find(sink, gate), gate, relative_path)

    def test_architecture_validation_precedes_weight_loading(self):
        for relative_path in (
                'modelscope/models/cv/image_view_transform/image_view_transform_infer.py',
                'modelscope/pipelines/cv/image_to_3d_pipeline.py'):
            source = self._source(relative_path)
            loader = source.index('def load_model')
            self.assertLess(
                source.index('instantiate_from_config', loader),
                source.index('torch.load', loader), relative_path)

        aec_source = self._source(
            'modelscope/pipelines/audio/linear_aec_pipeline.py')
        self.assertLess(
            aec_source.index('_validate_aec_config('),
            aec_source.index('self._init_model()'))

    def test_config_architectures_do_not_import_config_selected_modules(self):
        for relative_path in (
                'modelscope/models/cv/image_view_transform/util.py',
                'modelscope/models/cv/image_to_3d/ldm/util.py',
                'modelscope/models/cv/anydoor/ldm/util.py'):
            source = self._source(relative_path)
            self.assertNotIn('importlib.import_module(', source, relative_path)
            self.assertIn('instantiate_registered_architecture(', source,
                          relative_path)


if __name__ == '__main__':
    unittest.main()

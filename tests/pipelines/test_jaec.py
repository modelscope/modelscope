# Copyright (c) Alibaba, Inc. and its affiliates.

import ctypes
import io
import platform
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import json
import numpy as np
import soundfile as sf

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.models.audio.aec.jaec import JAECModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level


class _FakeFunction:

    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.function(*args)


class _FakeJAECLibrary:

    def __init__(self, create_handle=1, process_status=0, error=None):
        self.reset_count = 0
        self.process_count = 0
        self.destroy_count = 0
        self._create_handle = create_handle
        self._process_status = process_status
        self._error = error
        self.jaec_frontend_create = _FakeFunction(self._create)
        self.jaec_frontend_destroy = _FakeFunction(self._destroy)
        self.jaec_frontend_reset = _FakeFunction(self._reset)
        self.jaec_frontend_process = _FakeFunction(self._process)
        self.jaec_frontend_last_error = _FakeFunction(self._last_error)

    def _create(self, _):
        return self._create_handle

    def _destroy(self, _):
        self.destroy_count += 1

    def _reset(self, _):
        self.reset_count += 1

    def _process(self, _, mic, ref, sample_count, output):
        del ref
        self.process_count += 1
        if self._process_status:
            return self._process_status
        mic_data = np.ctypeslib.as_array(mic, shape=(sample_count, ))
        output_data = np.ctypeslib.as_array(output, shape=(sample_count, ))
        output_data[:] = mic_data
        return 0

    def _last_error(self):
        return self._error


class JAECPipelineTest(unittest.TestCase):

    @staticmethod
    @contextmanager
    def _mock_native(fake_native):
        with patch(
                'modelscope.models.audio.aec.jaec.ctypes.CDLL',
                return_value=fake_native), patch(
                    'modelscope.models.audio.aec.jaec.platform.system',
                    return_value='Linux'), patch(
                        'modelscope.models.audio.aec.jaec.platform.machine',
                        return_value='x86_64'):
            yield

    @staticmethod
    def _create_model_dir(root: Path):
        (root / 'lib').mkdir()
        (root / 'weights').mkdir()
        for library in ('jaec_arm.so', 'jaec_x86.so', 'jaec_x86.dll'):
            (root / 'lib' / library).touch()
        (root / 'weights' / 'tde_lp.bin').touch()
        (root / 'configuration.json').write_text(
            json.dumps({
                'framework': 'other',
                'task': 'acoustic-echo-cancellation',
                'pipeline': {
                    'type': 'speech-jaec-aec-16k'
                }
            }),
            encoding='utf-8')

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sample_rate=16000):
        sf.write(path, audio, sample_rate, subtype='PCM_16')

    def test_pipeline_processes_frames_and_resets(self):
        fake_native = _FakeJAECLibrary()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            mic = np.arange(321, dtype=np.int16) - 160
            ref = np.zeros(321, dtype=np.int16)
            mic_path = root / 'mic.wav'
            ref_path = root / 'ref.wav'
            output_path = root / 'output.wav'
            self._write_wav(mic_path, mic)
            self._write_wav(ref_path, ref)

            with self._mock_native(fake_native):
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model=str(root),
                    device='cpu',
                    trust_remote_code=True)
                inputs = {
                    'nearend_mic': str(mic_path),
                    'farend_speech': ref_path.read_bytes(),
                }
                first = aec(inputs, output_path=output_path)
                second = aec((mic, ref))
                aec.model.close()

            expected = mic.tobytes()
            self.assertEqual(first[OutputKeys.OUTPUT_PCM], expected)
            self.assertEqual(second[OutputKeys.OUTPUT_PCM], expected)
            written, sample_rate = sf.read(
                output_path, dtype='int16', always_2d=False)
            np.testing.assert_array_equal(written, mic)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(fake_native.reset_count, 2)
            self.assertEqual(fake_native.process_count, 6)
            self.assertEqual(fake_native.destroy_count, 1)

    def test_pipeline_rejects_misaligned_inputs(self):
        fake_native = _FakeJAECLibrary()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            mic_path = root / 'mic.wav'
            ref_path = root / 'ref.wav'
            self._write_wav(mic_path, np.zeros(320, dtype=np.int16))
            self._write_wav(ref_path, np.zeros(319, dtype=np.int16))

            with self._mock_native(fake_native):
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model=str(root),
                    device='cpu',
                    trust_remote_code=True)
                with self.assertRaisesRegex(ValueError, 'same length'):
                    aec({
                        'nearend_mic': str(mic_path),
                        'farend_speech': str(ref_path),
                    })
                aec.model.close()

    def test_model_rejects_unsupported_platform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            with patch(
                    'modelscope.models.audio.aec.jaec.platform.system',
                    return_value='Linux'), patch(
                        'modelscope.models.audio.aec.jaec.platform.machine',
                        return_value='aarch64'):
                with self.assertRaisesRegex(RuntimeError, 'supports macOS'):
                    JAECModel(str(root), trust_remote_code=True)

    def test_local_model_requires_explicit_trust(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            fake_native = _FakeJAECLibrary()
            with self._mock_native(fake_native):
                with self.assertRaisesRegex(RuntimeError,
                                            'trust_remote_code=True'):
                    pipeline(
                        Tasks.acoustic_echo_cancellation,
                        model=str(root),
                        device='cpu')
                with self.assertRaisesRegex(RuntimeError,
                                            'trust_remote_code=True'):
                    pipeline(
                        Tasks.acoustic_echo_cancellation,
                        model=str(root),
                        device='cpu',
                        trust_remote_code='false')
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model=str(root),
                    device='cpu',
                    trust_remote_code=True)
                aec.model.close()

    def test_iic_like_local_path_cannot_bypass_native_trust(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'iic' / 'speech_jaec_aec_16k'
            root.mkdir(parents=True)
            self._create_model_dir(root)
            fake_native = _FakeJAECLibrary()
            with self._mock_native(fake_native):
                with self.assertRaisesRegex(RuntimeError,
                                            'trust_remote_code=True'):
                    pipeline(
                        Tasks.acoustic_echo_cancellation,
                        model=str(root),
                        device='cpu')
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model=str(root),
                    device='cpu',
                    trust_remote_code=True)
                aec.model.close()

    def test_explicit_pipeline_downloads_requested_revision(self):
        fake_native = _FakeJAECLibrary()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            with self._mock_native(fake_native), patch(
                    'modelscope.pipelines.audio.jaec_pipeline.snapshot_download',
                    return_value=str(root)) as download:
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model='iic/speech_jaec_aec_16k',
                    pipeline_name=Pipelines.speech_jaec_aec_16k,
                    model_revision='v1.0.0',
                    device='cpu',
                    trust_remote_code=True)
                aec.model.close()
            download.assert_called_once_with(
                'iic/speech_jaec_aec_16k', revision='v1.0.0')

    def test_pipeline_rejects_invalid_audio_format(self):
        fake_native = _FakeJAECLibrary()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            mic_path = root / 'mic.wav'
            ref_path = root / 'ref.wav'
            self._write_wav(mic_path, np.zeros((320, 2), dtype=np.int16))
            self._write_wav(
                ref_path, np.zeros(320, dtype=np.int16), sample_rate=8000)

            with self._mock_native(fake_native):
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model=str(root),
                    device='cpu',
                    trust_remote_code=True)
                with self.assertRaisesRegex(ValueError, 'must be mono'):
                    aec({
                        'nearend_mic': str(mic_path),
                        'farend_speech': str(mic_path),
                    })
                with self.assertRaisesRegex(ValueError, '16 kHz'):
                    aec({
                        'nearend_mic': str(ref_path),
                        'farend_speech': str(ref_path),
                    })
                aec.model.close()

    def test_model_forward_preserves_frame_state(self):
        fake_native = _FakeJAECLibrary()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            frame = np.zeros(160, dtype=np.int16)
            with self._mock_native(fake_native):
                model = JAECModel(str(root), trust_remote_code=True)
                model({'mic': frame, 'ref': frame})
                model({'mic': frame, 'ref': frame})
                self.assertEqual(fake_native.reset_count, 0)
                self.assertEqual(fake_native.process_count, 2)
                model.reset()
                aec = pipeline(
                    Tasks.acoustic_echo_cancellation,
                    model=model,
                    pipeline_name=Pipelines.speech_jaec_aec_16k,
                    device='cpu')
                self.assertIs(aec.model, model)
                aec.model.close()
            self.assertEqual(fake_native.reset_count, 1)
            self.assertEqual(fake_native.destroy_count, 1)

    def test_native_errors_and_close_are_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            init_failure = _FakeJAECLibrary(
                create_handle=0, error=b'invalid weights')
            with self._mock_native(init_failure):
                with self.assertRaisesRegex(RuntimeError, 'invalid weights'):
                    JAECModel(str(root), trust_remote_code=True)

            process_failure = _FakeJAECLibrary(
                process_status=1, error=b'native process failed')
            frame = np.zeros(160, dtype=np.int16)
            with self._mock_native(process_failure):
                model = JAECModel(str(root), trust_remote_code=True)
                with self.assertRaisesRegex(RuntimeError,
                                            'native process failed'):
                    model.process(frame, frame)
                model.close()
                model.close()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    model.process(frame, frame)
            self.assertEqual(process_failure.destroy_count, 1)

    def test_missing_runtime_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_model_dir(root)
            (root / 'lib' / 'jaec_x86.so').unlink()
            with patch(
                    'modelscope.models.audio.aec.jaec.platform.system',
                    return_value='Linux'), patch(
                        'modelscope.models.audio.aec.jaec.platform.machine',
                        return_value='x86_64'):
                with self.assertRaisesRegex(RuntimeError,
                                            'native library not found'):
                    JAECModel(str(root), trust_remote_code=True)


@unittest.skipUnless(test_level() >= 1, 'skip network integration tests')
class JAECHubIntegrationTest(unittest.TestCase):
    """Exercise the published model and native library without mocks."""

    @classmethod
    def setUpClass(cls):
        supported = {
            'Darwin': ('arm64', 'aarch64'),
            'Linux': ('x86_64', 'amd64'),
            'Windows': ('x86_64', 'amd64'),
        }
        if platform.machine().lower() not in supported.get(
                platform.system(), ()):
            raise unittest.SkipTest(
                'the published JAEC runtime does not support this platform')

        cls.aec = pipeline(
            Tasks.acoustic_echo_cancellation,
            model='iic/speech_jaec_aec_16k',
            device='cpu',
            trust_remote_code=True)
        cls.addClassCleanup(cls.aec.model.close)

        sample_root = (
            'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/jaec')
        cls.urls = {
            'nearend_mic': f'{sample_root}/nearend_mic.wav',
            'farend_speech': f'{sample_root}/farend_speech.wav',
        }
        cls.wav_bytes = {
            name: File.read(url)
            for name, url in cls.urls.items()
        }
        cls.audio = {}
        for name, data in cls.wav_bytes.items():
            with sf.SoundFile(io.BytesIO(data)) as audio_file:
                if (audio_file.samplerate != 16000 or audio_file.channels != 1
                        or audio_file.subtype != 'PCM_16'):
                    raise AssertionError(
                        'JAEC samples must be 16 kHz mono PCM16 WAVs')
                cls.audio[name] = audio_file.read(dtype='int16')
        if (len(cls.audio['nearend_mic']) == 0 or len(cls.audio['nearend_mic'])
                != len(cls.audio['farend_speech'])):
            raise AssertionError(
                'JAEC samples must have equal non-zero length')

        temp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp_dir.cleanup)
        cls.root = Path(temp_dir.name)
        cls.wav_paths = {}
        for name, data in cls.wav_bytes.items():
            path = cls.root / f'{name}.wav'
            File.write(data, str(path))
            cls.wav_paths[name] = str(path)

    def _check_pcm(self, result):
        pcm = result[OutputKeys.OUTPUT_PCM]
        self.assertIsInstance(pcm, bytes)
        self.assertEqual(len(pcm), len(self.audio['nearend_mic']) * 2)
        output = np.frombuffer(pcm, dtype=np.int16)
        self.assertGreater(np.count_nonzero(output), 0)
        self.assertFalse(np.array_equal(output, self.audio['nearend_mic']))
        return pcm

    def test_public_model_with_urls_and_output_wav(self):
        output_path = self.root / 'output.wav'
        pcm = self._check_pcm(self.aec(self.urls, output_path=output_path))
        with sf.SoundFile(output_path) as output_file:
            self.assertEqual(output_file.samplerate, 16000)
            self.assertEqual(output_file.channels, 1)
            self.assertEqual(output_file.subtype, 'PCM_16')
            self.assertEqual(output_file.read(dtype='int16').tobytes(), pcm)

    def test_wav_paths_bytes_and_arrays_match(self):
        expected = self._check_pcm(self.aec(self.wav_paths))
        for name, inputs in (('bytes', self.wav_bytes), ('arrays',
                                                         self.audio)):
            with self.subTest(input_type=name):
                self.assertEqual(self._check_pcm(self.aec(inputs)), expected)
        self.assertEqual(self._check_pcm(self.aec(self.wav_paths)), expected)

    def test_partial_frame_matches_streaming_and_resets(self):
        samples = 1001
        self.assertGreaterEqual(len(self.audio['nearend_mic']), samples)
        mic = self.audio['nearend_mic'][:samples]
        ref = self.audio['farend_speech'][:samples]
        first = self.aec((mic, ref))[OutputKeys.OUTPUT_PCM]
        self.assertEqual(len(first), samples * 2)

        self.aec.model.reset()
        padding = (-samples) % 160
        mic_padded = np.pad(mic, (0, padding))
        ref_padded = np.pad(ref, (0, padding))
        frames = [
            self.aec.model.process(mic_padded[offset:offset + 160],
                                   ref_padded[offset:offset + 160])
            for offset in range(0, len(mic_padded), 160)
        ]
        expected = np.concatenate(frames)[:samples].tobytes()
        self.assertEqual(first, expected)
        self.assertEqual(self.aec((mic, ref))[OutputKeys.OUTPUT_PCM], expected)


if __name__ == '__main__':
    unittest.main()

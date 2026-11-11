# Copyright (c) Alibaba, Inc. and its affiliates.

import ctypes
import os
import platform
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np

from modelscope.models.base import Model


class JAECModel(Model):
    """Stateful native JAEC runtime loaded from a model snapshot."""

    def __init__(self,
                 model_dir: str,
                 *args,
                 trust_native_code: bool = False,
                 **kwargs):
        if trust_native_code is not True:
            raise RuntimeError(
                'JAEC loads a native library from the model repository. '
                'Pass trust_native_code=True only when you trust that '
                'repository.')
        kwargs.pop('device', None)
        super().__init__(model_dir, *args, device='cpu', **kwargs)

        system = platform.system()
        machine = platform.machine().lower()
        library_name = {
            ('Darwin', 'arm64'): 'jaec_arm.so',
            ('Darwin', 'aarch64'): 'jaec_arm.so',
            ('Linux', 'x86_64'): 'jaec_x86.so',
            ('Linux', 'amd64'): 'jaec_x86.so',
            ('Windows', 'x86_64'): 'jaec_x86.dll',
            ('Windows', 'amd64'): 'jaec_x86.dll',
        }.get((system, machine))
        if library_name is None:
            raise RuntimeError(
                'JAEC supports macOS arm64, Linux x86-64, and Windows '
                f'x86-64; got {system} {platform.machine()}.')

        model_root = Path(model_dir).resolve()
        library_path = (model_root / 'lib' / library_name).resolve()
        weights_path = (model_root / 'weights' / 'tde_lp.bin').resolve()
        for path, description in ((library_path, 'native library'),
                                  (weights_path, 'model weights')):
            if not path.is_relative_to(model_root) or not path.is_file():
                raise RuntimeError(f'JAEC {description} not found: {path}')

        self._lock = threading.RLock()
        self._handle = None
        try:
            self._native = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise RuntimeError(
                f'failed to load JAEC native library: {library_path}'
            ) from error

        pcm_pointer = ctypes.POINTER(ctypes.c_int16)
        self._native.jaec_frontend_create.argtypes = [ctypes.c_char_p]
        self._native.jaec_frontend_create.restype = ctypes.c_void_p
        self._native.jaec_frontend_destroy.argtypes = [ctypes.c_void_p]
        self._native.jaec_frontend_destroy.restype = None
        self._native.jaec_frontend_reset.argtypes = [ctypes.c_void_p]
        self._native.jaec_frontend_reset.restype = None
        self._native.jaec_frontend_process.argtypes = [
            ctypes.c_void_p,
            pcm_pointer,
            pcm_pointer,
            ctypes.c_int,
            pcm_pointer,
        ]
        self._native.jaec_frontend_process.restype = ctypes.c_int
        self._native.jaec_frontend_last_error.argtypes = []
        self._native.jaec_frontend_last_error.restype = ctypes.c_char_p

        self._handle = self._native.jaec_frontend_create(
            os.fsencode(weights_path))
        if not self._handle:
            raise RuntimeError(
                self._native_error('JAEC initialization failed'))

    def _native_error(self, fallback: str) -> str:
        error = self._native.jaec_frontend_last_error()
        return error.decode('utf-8', errors='replace') if error else fallback

    def _ensure_open(self):
        if not self._handle:
            raise RuntimeError('JAEC model is closed')

    def reset(self):
        with self._lock:
            self._ensure_open()
            self._native.jaec_frontend_reset(self._handle)

    def close(self):
        lock = getattr(self, '_lock', None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, '_handle', None)
            if handle:
                self._native.jaec_frontend_destroy(handle)
                self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _validate_frame(mic: np.ndarray, ref: np.ndarray):
        if not isinstance(mic, np.ndarray) or mic.dtype != np.int16:
            raise TypeError('mic must be a numpy.int16 array')
        if not isinstance(ref, np.ndarray) or ref.dtype != np.int16:
            raise TypeError('ref must be a numpy.int16 array')
        if mic.shape != (160, ) or ref.shape != (160, ):
            raise ValueError(
                'mic and ref must each contain one 160-sample frame')

    def _process_frame_unlocked(self, mic: np.ndarray,
                                ref: np.ndarray) -> np.ndarray:
        mic = np.ascontiguousarray(mic)
        ref = np.ascontiguousarray(ref)
        output = np.empty(160, dtype=np.int16)
        status = self._native.jaec_frontend_process(
            self._handle, mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), 160,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
        if status != 0:
            raise RuntimeError(self._native_error('JAEC processing failed'))
        return output

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Process one 160-sample frame while preserving stream state."""
        self._validate_frame(mic, ref)
        with self._lock:
            self._ensure_open()
            return self._process_frame_unlocked(mic, ref)

    def process_audio(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Reset once, then process an aligned utterance frame by frame."""
        if not isinstance(mic, np.ndarray) or mic.dtype != np.int16:
            raise TypeError('mic must be a numpy.int16 array')
        if not isinstance(ref, np.ndarray) or ref.dtype != np.int16:
            raise TypeError('ref must be a numpy.int16 array')
        if mic.ndim != 1 or ref.ndim != 1:
            raise ValueError('mic and ref must be one-dimensional')
        if len(mic) == 0 or len(mic) != len(ref) or len(mic) % 160:
            raise ValueError(
                'mic and ref must have the same non-zero length, padded to '
                'a multiple of 160 samples')

        output = np.empty(len(mic), dtype=np.int16)
        with self._lock:
            self._ensure_open()
            self._native.jaec_frontend_reset(self._handle)
            for offset in range(0, len(mic), 160):
                output[offset:offset + 160] = self._process_frame_unlocked(
                    mic[offset:offset + 160], ref[offset:offset + 160])
        return output

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if not isinstance(inputs, dict):
            raise TypeError('JAEC model input must be a dictionary')
        return {'output': self.process(inputs.get('mic'), inputs.get('ref'))}

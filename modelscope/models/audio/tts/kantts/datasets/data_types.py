# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from scipy.io import wavfile

DATA_TYPE_DICT = {
    'txt': {
        'load_func': np.loadtxt,
        'desc': 'plain txt file or readable by np.loadtxt',
    },
    'wav': {
        'load_func': lambda x: wavfile.read(x)[1],
        'desc': 'wav file or readable by soundfile.read',
    },
    'npy': {
        'load_func': np.load,
        'desc': 'any .npy format file',
    },
    # PCM data type can be loaded by binary format
    'bin_f32': {
        'load_func': lambda x: np.fromfile(x, dtype=np.float32),
        'desc': 'binary file with float32 format',
    },
    'bin_f64': {
        'load_func': lambda x: np.fromfile(x, dtype=np.float64),
        'desc': 'binary file with float64 format',
    },
    'bin_i32': {
        'load_func': lambda x: np.fromfile(x, dtype=np.int32),
        'desc': 'binary file with int32 format',
    },
    'bin_i16': {
        'load_func': lambda x: np.fromfile(x, dtype=np.int16),
        'desc': 'binary file with int16 format',
    },
}

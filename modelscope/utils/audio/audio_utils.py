# Copyright (c) Alibaba, Inc. and its affiliates.
SEGMENT_LENGTH_TRAIN = 16000


def to_segment(batch, segment_length=SEGMENT_LENGTH_TRAIN):
    """
    Dataset mapping function to split one audio into segments.
    It only works in batch mode.
    """
    noisy_arrays = []
    clean_arrays = []
    for x, y in zip(batch['noisy'], batch['clean']):
        length = min(len(x['array']), len(y['array']))
        noisy = x['array']
        clean = y['array']
        for offset in range(segment_length, length + 1, segment_length):
            noisy_arrays.append(noisy[offset - segment_length:offset])
            clean_arrays.append(clean[offset - segment_length:offset])
    return {'noisy': noisy_arrays, 'clean': clean_arrays}


def audio_norm(x):
    rms = (x**2).mean()**0.5
    scalar = 10**(-25 / 20) / rms
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean()**0.5
    scalarx = 10**(-25 / 20) / rmsx
    x = x * scalarx
    return x

# Copyright (c) Alibaba, Inc. and its affiliates.
import re
import struct
from typing import Union
from urllib.parse import urlparse

from modelscope.fileio.file import HTTPStorage

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


def update_conf(origin_config_file, new_config_file, conf_item: [str, str]):

    def repl(matched):
        key = matched.group(1)
        if key in conf_item:
            return conf_item[key]
        else:
            return None

    with open(origin_config_file, encoding='utf-8') as f:
        lines = f.readlines()
    with open(new_config_file, 'w') as f:
        for line in lines:
            line = re.sub(r'\$\{(.*)\}', repl, line)
            f.write(line)


def extract_pcm_from_wav(wav: bytes) -> bytes:
    data = wav
    sample_rate = None
    if len(data) > 44:
        frame_len = 44
        file_len = len(data)
        try:
            header_fields = {}
            header_fields['ChunkID'] = str(data[0:4], 'UTF-8')
            header_fields['Format'] = str(data[8:12], 'UTF-8')
            header_fields['Subchunk1ID'] = str(data[12:16], 'UTF-8')
            if header_fields['ChunkID'] == 'RIFF' and header_fields[
                    'Format'] == 'WAVE' and header_fields[
                        'Subchunk1ID'] == 'fmt ':
                header_fields['SubChunk1Size'] = struct.unpack(
                    '<I', data[16:20])[0]
                header_fields['SampleRate'] = struct.unpack('<I',
                                                            data[24:28])[0]
                sample_rate = header_fields['SampleRate']

                if header_fields['SubChunk1Size'] == 16:
                    frame_len = 44
                elif header_fields['SubChunk1Size'] == 18:
                    frame_len = 46
                else:
                    return data, sample_rate

                data = wav[frame_len:file_len]
        except Exception:
            # no treatment
            pass

    return data, sample_rate


def load_bytes_from_url(url: str) -> Union[bytes, str]:
    sample_rate = None
    result = urlparse(url)
    if result.scheme is not None and len(result.scheme) > 0:
        storage = HTTPStorage()
        data = storage.read(url)
        data, sample_rate = extract_pcm_from_wav(data)
    else:
        data = url

    return data, sample_rate

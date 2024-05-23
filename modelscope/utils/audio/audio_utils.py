# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import struct
import sys
import tempfile
from typing import Union
from urllib.parse import urlparse

import numpy as np

from modelscope.fileio.file import HTTPStorage
from modelscope.utils.file_utils import get_model_cache_root
from modelscope.utils.hub import snapshot_download
from modelscope.utils.logger import get_logger

logger = get_logger()

SEGMENT_LENGTH_TRAIN = 16000
SUPPORT_AUDIO_TYPE_SETS = ('.flac', '.mp3', '.ogg', '.opus', '.wav', '.pcm')


class TtsTrainType(object):
    TRAIN_TYPE_SAMBERT = 'train-type-sambert'
    TRAIN_TYPE_BERT = 'train-type-bert'
    TRAIN_TYPE_VOC = 'train-type-voc'


class TtsCustomParams(object):
    VOICE_NAME = 'voice_name'
    AM_CKPT = 'am_ckpt'
    VOC_CKPT = 'voc_ckpt'
    AM_CONFIG = 'am_config'
    VOC_CONFIG = 'voc_config'
    AUIDO_CONFIG = 'audio_config'
    SE_FILE = 'se_file'
    SE_MODEL = 'se_model'
    MVN_FILE = 'mvn_file'


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
            value = conf_item[key]
            if not isinstance(value, str):
                value = str(value)
            return value
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


def expect_token_number(instr, token):
    first_token = re.match(r'^\s*' + token, instr)
    if first_token is None:
        return None
    instr = instr[first_token.end():]
    lr = re.match(r'^\s*(-?\d+\.?\d*e?-?\d*?)', instr)
    if lr is None:
        return None
    return instr[lr.end():], lr.groups()[0]


def expect_kaldi_matrix(instr):
    pos2 = instr.find('[', 0)
    pos3 = instr.find(']', pos2)
    mat = []
    for stt in instr[pos2 + 1:pos3].split('\n'):
        tmp_mat = np.fromstring(stt, dtype=np.float32, sep=' ')
        if tmp_mat.size > 0:
            mat.append(tmp_mat)
    return instr[pos3 + 1:], np.array(mat)


# This implementation is adopted from scipy.io.wavfile.write,
# made publicly available under the BSD-3-Clause license at
# https://github.com/scipy/scipy/blob/v1.9.3/scipy/io/wavfile.py
def ndarray_pcm_to_wav(fs: int, data: np.ndarray) -> bytes:
    dkind = data.dtype.kind
    if not (dkind == 'i' or dkind == 'f' or  # noqa W504
            (dkind == 'u' and data.dtype.itemsize == 1)):
        raise ValueError(f'Unsupported data type {data.dtype}')

    header_data = bytearray()
    header_data += b'RIFF'
    header_data += b'\x00\x00\x00\x00'
    header_data += b'WAVE'
    header_data += b'fmt '
    if dkind == 'f':
        format_tag = 0x0003
    else:
        format_tag = 0x0001
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]
    bit_depth = data.dtype.itemsize * 8
    bytes_per_second = fs * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)

    fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs,
                                 bytes_per_second, block_align, bit_depth)
    if not (dkind == 'i' or dkind == 'u'):
        fmt_chunk_data += b'\x00\x00'
    header_data += struct.pack('<I', len(fmt_chunk_data))
    header_data += fmt_chunk_data

    if not (dkind == 'i' or dkind == 'u'):
        header_data += b'fact'
        header_data += struct.pack('<II', 4, data.shape[0])

    if ((len(header_data) - 8) + (8 + data.nbytes)) > 0xFFFFFFFF:
        raise ValueError('Data exceeds wave file size limit')

    header_data += b'data'
    header_data += struct.pack('<I', data.nbytes)
    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '='
                                       and sys.byteorder == 'big'):
        data = data.byteswap()
    header_data += data.ravel().view('b').data
    size = len(header_data)
    header_data[4:8] = struct.pack('<I', size - 8)
    return bytes(header_data)


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


def generate_scp_from_url(url: str, key: str = None):
    wav_scp_path = None
    raw_inputs = None
    # for local inputs
    if os.path.exists(url):
        wav_scp_path = url
        return wav_scp_path, raw_inputs
    # for wav url, download bytes data
    if url.startswith('http'):
        result = urlparse(url)
        if result.scheme is not None and len(result.scheme) > 0:
            storage = HTTPStorage()
            # bytes
            data = storage.read(url)
            work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            wav_path = os.path.join(work_dir, os.path.basename(url))
            with open(wav_path, 'wb') as fb:
                fb.write(data)
            return wav_path, raw_inputs

    return wav_scp_path, raw_inputs


def generate_text_from_url(url: str):
    text_file_path = None
    raw_inputs = None
    # for text str input
    if not os.path.exists(url) and not url.startswith('http'):
        raw_inputs = url
        return text_file_path, raw_inputs

    # for local txt inputs
    if os.path.exists(url) and (url.lower().endswith('.txt')
                                or url.lower().endswith('.scp')):
        text_file_path = url
        return text_file_path, raw_inputs
    # for url, download and generate txt
    result = urlparse(url)
    if result.scheme is not None and len(result.scheme) > 0:
        storage = HTTPStorage()
        data = storage.read(url)
        work_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        text_file_path = os.path.join(work_dir, os.path.basename(url))
        with open(text_file_path, 'wb') as fp:
            fp.write(data)
        return text_file_path, raw_inputs

    return text_file_path, raw_inputs


def generate_scp_for_sv(url: str, key: str = None):
    wav_scp_path = None
    wav_name = key if key is not None else os.path.basename(url)
    # for local wav.scp inputs
    if os.path.exists(url) and url.lower().endswith('.scp'):
        wav_scp_path = url
        return wav_scp_path
    # for local wav file inputs
    if os.path.exists(url) and (url.lower().endswith(SUPPORT_AUDIO_TYPE_SETS)):
        wav_path = url
        work_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        wav_scp_path = os.path.join(work_dir, 'wav.scp')
        with open(wav_scp_path, 'w') as ft:
            scp_content = '\t'.join([wav_name, wav_path]) + '\n'
            ft.writelines(scp_content)
        return wav_scp_path
    # for wav url, download and generate wav.scp
    result = urlparse(url)
    if result.scheme is not None and len(result.scheme) > 0:
        storage = HTTPStorage()
        wav_scp_path = storage.read(url)
        return wav_scp_path

    return wav_scp_path


def generate_sv_scp_from_url(urls: Union[tuple, list]):
    """
    generate audio_scp files from url input for speaker verification.
    """
    audio_scps = []
    for url in urls:
        audio_scp = generate_scp_for_sv(url, key='test1')
        audio_scps.append(audio_scp)
    return audio_scps


def generate_sd_scp_from_url(urls: Union[tuple, list]):
    """
    generate audio_scp files from url input for speaker diarization.
    """
    audio_scps = []
    for url in urls:
        if os.path.exists(url) and (
                url.lower().endswith(SUPPORT_AUDIO_TYPE_SETS)):
            audio_scp = url
        else:
            result = urlparse(url)
            if result.scheme is not None and len(result.scheme) > 0:
                storage = HTTPStorage()
                wav_bytes = storage.read(url)
                audio_scp = wav_bytes
            else:
                raise ValueError("Can't download from {}.".format(url))
        audio_scps.append(audio_scp)
    return audio_scps


def update_local_model(model_config, model_path, extra_args):
    if 'update_model' in extra_args and not extra_args['update_model']:
        return
    model_revision = None
    if 'update_model' in extra_args:
        if extra_args['update_model'] == 'latest':
            model_revision = None
        else:
            model_revision = extra_args['update_model']
    if model_config.__contains__('model'):
        model_name = model_config['model']
        dst_dir_root = get_model_cache_root()
        if isinstance(model_path, str) and os.path.exists(
                model_path) and not model_path.startswith(dst_dir_root):
            try:
                dst = os.path.join(dst_dir_root, '.cache/' + model_name)
                dst_dir = os.path.dirname(dst)
                os.makedirs(dst_dir, exist_ok=True)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(model_path), dst)

                snapshot_download(
                    model_name,
                    cache_dir=dst_dir_root,
                    revision=model_revision)
            except Exception as e:
                logger.warning(str(e))
    else:
        logger.warning('Can not find model name in configuration')

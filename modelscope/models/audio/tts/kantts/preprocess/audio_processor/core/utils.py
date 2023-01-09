# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import librosa
import numpy as np
import pysptk
import sox
from scipy.io import wavfile
from tqdm import tqdm

from modelscope.utils.logger import get_logger
from .dsp import _stft

logging = get_logger()

anchor_hist = np.array([
    0.0,
    0.00215827,
    0.00354383,
    0.00442313,
    0.00490274,
    0.00532907,
    0.00602185,
    0.00690115,
    0.00810019,
    0.00948574,
    0.0120437,
    0.01489475,
    0.01873168,
    0.02302158,
    0.02872369,
    0.03669065,
    0.04636291,
    0.05843325,
    0.07700506,
    0.11052491,
    0.16802558,
    0.25997868,
    0.37942979,
    0.50730083,
    0.62006395,
    0.71092459,
    0.76877165,
    0.80762057,
    0.83458566,
    0.85672795,
    0.87660538,
    0.89251266,
    0.90578204,
    0.91569411,
    0.92541966,
    0.93383959,
    0.94162004,
    0.94940048,
    0.95539568,
    0.96136424,
    0.9670397,
    0.97290168,
    0.97705835,
    0.98116174,
    0.98465228,
    0.98814282,
    0.99152678,
    0.99421796,
    0.9965894,
    0.99840128,
    1.0,
])

anchor_bins = np.array([
    0.033976,
    0.03529014,
    0.03660428,
    0.03791842,
    0.03923256,
    0.0405467,
    0.04186084,
    0.04317498,
    0.04448912,
    0.04580326,
    0.0471174,
    0.04843154,
    0.04974568,
    0.05105982,
    0.05237396,
    0.0536881,
    0.05500224,
    0.05631638,
    0.05763052,
    0.05894466,
    0.0602588,
    0.06157294,
    0.06288708,
    0.06420122,
    0.06551536,
    0.0668295,
    0.06814364,
    0.06945778,
    0.07077192,
    0.07208606,
    0.0734002,
    0.07471434,
    0.07602848,
    0.07734262,
    0.07865676,
    0.0799709,
    0.08128504,
    0.08259918,
    0.08391332,
    0.08522746,
    0.0865416,
    0.08785574,
    0.08916988,
    0.09048402,
    0.09179816,
    0.0931123,
    0.09442644,
    0.09574058,
    0.09705472,
    0.09836886,
    0.099683,
])

hist_bins = 50


def amp_info(wav_file_path):
    """
    Returns the amplitude info of the wav file.
    """
    stats = sox.file_info.stat(wav_file_path)
    amp_rms = stats['RMS     amplitude']
    amp_max = stats['Maximum amplitude']
    amp_mean = stats['Mean    amplitude']
    length = stats['Length (seconds)']

    return {
        'amp_rms': amp_rms,
        'amp_max': amp_max,
        'amp_mean': amp_mean,
        'length': length,
        'basename': os.path.basename(wav_file_path),
    }


def statistic_amplitude(src_wav_dir):
    """
    Returns the amplitude info of the wav file.
    """
    wav_lst = glob(os.path.join(src_wav_dir, '*.wav'))
    with ProcessPoolExecutor(max_workers=8) as executor, tqdm(
            total=len(wav_lst)) as progress:
        futures = []
        for wav_file_path in wav_lst:
            future = executor.submit(amp_info, wav_file_path)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        amp_info_lst = [future.result() for future in futures]

    amp_info_lst = sorted(amp_info_lst, key=lambda x: x['amp_rms'])

    logging.info('Average amplitude RMS : {}'.format(
        np.mean([x['amp_rms'] for x in amp_info_lst])))

    return amp_info_lst


def volume_normalize(src_wav_dir, out_wav_dir):
    logging.info('Volume statistic proceeding...')
    amp_info_lst = statistic_amplitude(src_wav_dir)
    logging.info('Volume statistic done.')

    rms_amp_lst = [x['amp_rms'] for x in amp_info_lst]
    src_hist, src_bins = np.histogram(
        rms_amp_lst, bins=hist_bins, density=True)
    src_hist = src_hist / np.sum(src_hist)
    src_hist = np.cumsum(src_hist)
    src_hist = np.insert(src_hist, 0, 0.0)

    logging.info('Volume normalization proceeding...')
    for amp_info in tqdm(amp_info_lst):
        rms_amp = amp_info['amp_rms']
        rms_amp = np.clip(rms_amp, src_bins[0], src_bins[-1])

        src_idx = np.where(rms_amp >= src_bins)[0][-1]
        src_pos = src_hist[src_idx]
        anchor_idx = np.where(src_pos >= anchor_hist)[0][-1]

        if src_idx == hist_bins or anchor_idx == hist_bins:
            rms_amp = anchor_bins[-1]
        else:
            rms_amp = (rms_amp - src_bins[src_idx]) / (
                src_bins[src_idx + 1] - src_bins[src_idx]) * (
                    anchor_bins[anchor_idx + 1]
                    - anchor_bins[anchor_idx]) + anchor_bins[anchor_idx]

        scale = rms_amp / amp_info['amp_rms']

        # FIXME: This is a hack to avoid the sound cliping.
        sr, data = wavfile.read(
            os.path.join(src_wav_dir, amp_info['basename']))
        wavfile.write(
            os.path.join(out_wav_dir, amp_info['basename']),
            sr,
            (data * scale).astype(np.int16),
        )
    logging.info('Volume normalization done.')

    return True


def interp_f0(f0_data):
    """
    linear interpolation
    """
    f0_data[f0_data < 1] = 0
    xp = np.nonzero(f0_data)
    yp = f0_data[xp]
    x = np.arange(f0_data.size)
    contour_f0 = np.interp(x, xp[0], yp).astype(np.float32)
    return contour_f0


def frame_nccf(x, y):
    norm_coef = (np.sum(x**2.0) * np.sum(y**2.0) + 1e-30)**0.5
    return (np.sum(x * y) / norm_coef + 1.0) / 2.0


def get_nccf(pcm_data, f0, min_f0=40, max_f0=800, fs=160, sr=16000):
    if pcm_data.dtype == np.int16:
        pcm_data = pcm_data.astype(np.float32) / 32768
    frame_len = int(sr / 200)
    frame_num = int(len(pcm_data) // fs)
    frame_num = min(frame_num, len(f0))

    pad_len = int(sr / min_f0) + frame_len

    pad_zeros = np.zeros([pad_len], dtype=np.float32)
    data = np.hstack((pad_zeros, pcm_data.astype(np.float32), pad_zeros))

    nccf = np.zeros((frame_num), dtype=np.float32)

    for i in range(frame_num):
        curr_f0 = np.clip(f0[i], min_f0, max_f0)
        lag = int(sr / curr_f0 + 0.5)
        j = i * fs + pad_len - frame_len // 2

        l_data = data[j:j + frame_len]
        l_data -= l_data.mean()

        r_data = data[j + lag:j + lag + frame_len]
        r_data -= r_data.mean()

        nccf[i] = frame_nccf(l_data, r_data)

    return nccf


def smooth(data, win_len):
    if win_len % 2 == 0:
        win_len += 1
    hwin = win_len // 2
    win = np.hanning(win_len)
    win /= win.sum()
    data = data.reshape([-1])
    pad_data = np.pad(data, hwin, mode='edge')

    for i in range(data.shape[0]):
        data[i] = np.dot(win, pad_data[i:i + win_len])

    return data.reshape([-1, 1])


#  support: rapt, swipe
#  unsupport: reaper, world(DIO)
def RAPT_FUNC(v1, v2, v3, v4, v5):
    return pysptk.sptk.rapt(
        v1.astype(np.float32), fs=v2, hopsize=v3, min=v4, max=v5)


def SWIPE_FUNC(v1, v2, v3, v4, v5):
    return pysptk.sptk.swipe(
        v1.astype(np.float64), fs=v2, hopsize=v3, min=v4, max=v5)


def PYIN_FUNC(v1, v2, v3, v4, v5):
    f0_mel = librosa.pyin(
        v1.astype(np.float32), sr=v2, frame_length=v3 * 4, fmin=v4, fmax=v5)[0]
    f0_mel = np.where(np.isnan(f0_mel), 0.0, f0_mel)
    return f0_mel


def get_pitch(pcm_data, sampling_rate=16000, hop_length=160):
    log_f0_list = []
    uv_list = []
    low, high = 40, 800

    cali_f0 = pysptk.sptk.rapt(
        pcm_data.astype(np.float32),
        fs=sampling_rate,
        hopsize=hop_length,
        min=low,
        max=high,
    )
    f0_range = np.sort(np.unique(cali_f0))

    if len(f0_range) > 20:
        low = max(f0_range[10] - 50, low)
        high = min(f0_range[-10] + 50, high)

    func_dict = {'rapt': RAPT_FUNC, 'swipe': SWIPE_FUNC}

    for func_name in func_dict:
        f0 = func_dict[func_name](pcm_data, sampling_rate, hop_length, low,
                                  high)
        uv = f0 > 0

        if len(f0) < 10 or f0.max() < low:
            logging.error('{} method: calc F0 is too low.'.format(func_name))
            continue
        else:
            f0 = np.clip(f0, 1e-30, high)
            log_f0 = np.log(f0)
            contour_log_f0 = interp_f0(log_f0)

            log_f0_list.append(contour_log_f0)
            uv_list.append(uv)

    if len(log_f0_list) == 0:
        logging.error('F0 estimation failed.')
        return None

    min_len = float('inf')
    for log_f0 in log_f0_list:
        min_len = min(min_len, log_f0.shape[0])

    multi_log_f0 = np.zeros([len(log_f0_list), min_len], dtype=np.float32)
    multi_uv = np.zeros([len(log_f0_list), min_len], dtype=np.float32)

    for i in range(len(log_f0_list)):
        multi_log_f0[i, :] = log_f0_list[i][:min_len]
        multi_uv[i, :] = uv_list[i][:min_len]

    log_f0 = smooth(np.median(multi_log_f0, axis=0), 5)
    uv = (smooth(np.median(multi_uv, axis=0), 5) > 0.5).astype(np.float32)

    f0 = np.exp(log_f0)

    min_len = min(f0.shape[0], uv.shape[0])

    return f0[:min_len], uv[:min_len], f0[:min_len] * uv[:min_len]


def get_energy(pcm_data, hop_length, win_length, n_fft):
    D = _stft(pcm_data, hop_length, win_length, n_fft)
    S, _ = librosa.magphase(D)
    energy = np.sqrt(np.sum(S**2, axis=0))

    return energy.reshape((-1, 1))


def align_length(in_data, tgt_data, basename=None):
    if in_data is None or tgt_data is None:
        logging.error('{}: Input data is None.'.format(basename))
        return None
    in_len = in_data.shape[0]
    tgt_len = tgt_data.shape[0]
    if abs(in_len - tgt_len) > 20:
        logging.error(
            '{}: Input data length mismatches with target data length too much.'
            .format(basename))
        return None

    if in_len < tgt_len:
        out_data = np.pad(
            in_data, ((0, tgt_len - in_len), (0, 0)),
            'constant',
            constant_values=0.0)
    else:
        out_data = in_data[:tgt_len]

    return out_data


def compute_mean(data_list, dims=80):
    mean_vector = np.zeros((1, dims))
    all_frame_number = 0

    for data in tqdm(data_list):
        if data is None:
            continue
        features = data.reshape((-1, dims))
        current_frame_number = np.shape(features)[0]
        mean_vector += np.sum(features[:, :], axis=0)
        all_frame_number += current_frame_number

    mean_vector /= float(all_frame_number)
    return mean_vector


def compute_std(data_list, mean_vector, dims=80):
    std_vector = np.zeros((1, dims))
    all_frame_number = 0

    for data in tqdm(data_list):
        if data is None:
            continue
        features = data.reshape((-1, dims))
        current_frame_number = np.shape(features)[0]
        mean_matrix = np.tile(mean_vector, (current_frame_number, 1))
        std_vector += np.sum((features[:, :] - mean_matrix)**2, axis=0)
        all_frame_number += current_frame_number

    std_vector /= float(all_frame_number)
    std_vector = std_vector**0.5
    return std_vector


F0_MIN = 0.0
F0_MAX = 800.0

ENERGY_MIN = 0.0
ENERGY_MAX = 200.0

CLIP_FLOOR = 1e-3


def f0_norm_min_max(f0):
    zero_idxs = np.where(f0 <= CLIP_FLOOR)[0]
    res = (2 * f0 - F0_MIN - F0_MAX) / (F0_MAX - F0_MIN)
    res[zero_idxs] = 0.0
    return res


def f0_denorm_min_max(f0):
    zero_idxs = np.where(f0 == 0.0)[0]
    res = (f0 * (F0_MAX - F0_MIN) + F0_MIN + F0_MAX) / 2
    res[zero_idxs] = 0.0
    return res


def energy_norm_min_max(energy):
    zero_idxs = np.where(energy == 0.0)[0]
    res = (2 * energy - ENERGY_MIN - ENERGY_MAX) / (ENERGY_MAX - ENERGY_MIN)
    res[zero_idxs] = 0.0
    return res


def energy_denorm_min_max(energy):
    zero_idxs = np.where(energy == 0.0)[0]
    res = (energy * (ENERGY_MAX - ENERGY_MIN) + ENERGY_MIN + ENERGY_MAX) / 2
    res[zero_idxs] = 0.0
    return res


def norm_log(x):
    zero_idxs = np.where(x <= CLIP_FLOOR)[0]
    x[zero_idxs] = 1.0
    res = np.log(x)
    return res


def denorm_log(x):
    zero_idxs = np.where(x == 0.0)[0]
    res = np.exp(x)
    res[zero_idxs] = 0.0
    return res


def f0_norm_mean_std(x, mean, std):
    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x


def norm_mean_std(x, mean, std):
    x = (x - mean) / std
    return x


def parse_interval_file(file_path, sampling_rate, hop_length):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    #  second
    frame_intervals = 1.0 * hop_length / sampling_rate
    skip_lines = 12
    dur_list = []
    phone_list = []

    line_index = skip_lines

    while line_index < len(lines):
        phone_begin = float(lines[line_index])
        phone_end = float(lines[line_index + 1])
        phone = lines[line_index + 2].strip()[1:-1]
        dur_list.append(
            int(round((phone_end - phone_begin) / frame_intervals)))
        phone_list.append(phone)
        line_index += 3

    if len(dur_list) == 0 or len(phone_list) == 0:
        return None

    return np.array(dur_list), phone_list


def average_by_duration(x, durs):
    if x is None or durs is None:
        return None
    durs_cum = np.cumsum(np.pad(durs, (1, 0), 'constant'))

    # average over each symbol's duraion
    x_symbol = np.zeros((durs.shape[0], ), dtype=np.float32)
    for idx, start, end in zip(
            range(durs.shape[0]), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_symbol[idx] = np.mean(values) if len(values) > 0 else 0.0

    return x_symbol.astype(np.float32)


def encode_16bits(x):
    if x.min() > -1.0 and x.max() < 1.0:
        return np.clip(x * 2**15, -(2**15), 2**15 - 1).astype(np.int16)
    else:
        return x

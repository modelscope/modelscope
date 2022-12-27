# Copyright (c) Alibaba, Inc. and its affiliates.

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


def _stft(y, hop_length, win_length, n_fft):
    return librosa.stft(
        y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    if wav.dtype == np.float32 or wav.dtype == np.float64:
        quant_wav = 32767 * wav
    else:
        quant_wav = wav
        # maxmize the volume to avoid clipping
        # wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, quant_wav.astype(np.int16))


def trim_silence(wav, top_db, hop_length, win_length):
    trimed_wav, _ = librosa.effects.trim(
        wav, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    return trimed_wav


def trim_silence_with_interval(wav, interval, hop_length):
    if interval is None:
        return None
    leading_sil = interval[0]
    tailing_sil = interval[-1]
    trim_wav = wav[leading_sil * hop_length:-tailing_sil * hop_length]
    return trim_wav


def preemphasis(wav, k=0.98, preemphasize=False):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k=0.98, inv_preemphasize=False):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def _normalize(S, max_norm=1.0, min_level_db=-100, symmetric=False):
    if symmetric:
        return np.clip(
            (2 * max_norm) * ((S - min_level_db) / (-min_level_db)) - max_norm,
            -max_norm,
            max_norm,
        )
    else:
        return np.clip(max_norm * ((S - min_level_db) / (-min_level_db)), 0,
                       max_norm)


def _denormalize(D, max_norm=1.0, min_level_db=-100, symmetric=False):
    if symmetric:
        return ((np.clip(D, -max_norm, max_norm) + max_norm) * -min_level_db
                /  # noqa W504
                (2 * max_norm)) + min_level_db
    else:
        return (np.clip(D, 0, max_norm) * -min_level_db
                / max_norm) + min_level_db


def _griffin_lim(S, n_fft, hop_length, win_length, griffin_lim_iters=60):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(
        S_complex * angles, hop_length=hop_length, win_length=win_length)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(
            _stft(
                y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = _istft(
            S_complex * angles, hop_length=hop_length, win_length=win_length)
    return y


def spectrogram(
    y,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    max_norm=1.0,
    min_level_db=-100,
    ref_level_db=20,
    symmetric=False,
):
    D = _stft(preemphasis(y), hop_length, win_length, n_fft)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S, max_norm, min_level_db, symmetric)


def inv_spectrogram(
    spectrogram,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    max_norm=1.0,
    min_level_db=-100,
    ref_level_db=20,
    symmetric=False,
    power=1.5,
):
    S = _db_to_amp(
        _denormalize(spectrogram, max_norm, min_level_db, symmetric)
        + ref_level_db)
    return _griffin_lim(S**power, n_fft, hop_length, win_length)


def _build_mel_basis(sample_rate, n_fft=1024, fmin=50, fmax=8000, n_mels=80):
    assert fmax <= sample_rate // 2
    return librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


# mel linear Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram,
                   sample_rate,
                   n_fft=1024,
                   fmin=50,
                   fmax=8000,
                   n_mels=80):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(sample_rate, n_fft, fmin, fmax, n_mels)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram,
                   sample_rate,
                   n_fft=1024,
                   fmin=50,
                   fmax=8000,
                   n_mels=80):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(
            _build_mel_basis(sample_rate, n_fft, fmin, fmax, n_mels))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def melspectrogram(
    y,
    sample_rate,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    max_norm=1.0,
    min_level_db=-100,
    ref_level_db=20,
    fmin=50,
    fmax=8000,
    symmetric=False,
    preemphasize=False,
):
    D = _stft(
        preemphasis(y, preemphasize=preemphasize),
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    )
    S = (
        _amp_to_db(
            _linear_to_mel(
                np.abs(D),
                sample_rate=sample_rate,
                n_fft=n_fft,
                fmin=fmin,
                fmax=fmax,
                n_mels=n_mels,
            )) - ref_level_db)
    return _normalize(
        S, max_norm=max_norm, min_level_db=min_level_db, symmetric=symmetric).T


def inv_mel_spectrogram(
    mel_spectrogram,
    sample_rate,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    max_norm=1.0,
    min_level_db=-100,
    ref_level_db=20,
    fmin=50,
    fmax=8000,
    power=1.5,
    symmetric=False,
    preemphasize=False,
):
    D = _denormalize(
        mel_spectrogram,
        max_norm=max_norm,
        min_level_db=min_level_db,
        symmetric=symmetric,
    )
    S = _mel_to_linear(
        _db_to_amp(D + ref_level_db),
        sample_rate=sample_rate,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        n_mels=n_mels,
    )
    return inv_preemphasis(
        _griffin_lim(S**power, n_fft, hop_length, win_length),
        preemphasize=preemphasize,
    )

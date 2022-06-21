import ctypes
import os
from typing import Any, Dict

import numpy as np
import scipy.io.wavfile as wav
import torch
from numpy.ctypeslib import ndpointer

from modelscope.utils.constant import Fields
from .builder import PREPROCESSORS


def load_wav(path):
    samp_rate, data = wav.read(path)
    return np.float32(data), samp_rate


def load_library(libaec):
    libaec_in_cwd = os.path.join('.', libaec)
    if os.path.exists(libaec_in_cwd):
        libaec = libaec_in_cwd
    mitaec = ctypes.cdll.LoadLibrary(libaec)
    fe_process = mitaec.fe_process_inst
    fe_process.argtypes = [
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'), ctypes.c_int,
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS')
    ]
    return fe_process


def do_linear_aec(fe_process, mic, ref, int16range=True):
    mic = np.float32(mic)
    ref = np.float32(ref)
    if len(mic) > len(ref):
        mic = mic[:len(ref)]
    out_mic = np.zeros_like(mic)
    out_linear = np.zeros_like(mic)
    out_echo = np.zeros_like(mic)
    out_ref = np.zeros_like(mic)
    if int16range:
        mic /= 32768
        ref /= 32768
    fe_process(mic, ref, len(mic), out_mic, out_linear, out_echo)
    # out_ref not in use here
    if int16range:
        out_mic *= 32768
        out_linear *= 32768
        out_echo *= 32768
    return out_mic, out_ref, out_linear, out_echo


def load_kaldi_feature_transform(filename):
    fp = open(filename, 'r')
    all_str = fp.read()
    pos1 = all_str.find('AddShift')
    pos2 = all_str.find('[', pos1)
    pos3 = all_str.find(']', pos2)
    mean = np.fromstring(all_str[pos2 + 1:pos3], dtype=np.float32, sep=' ')
    pos1 = all_str.find('Rescale')
    pos2 = all_str.find('[', pos1)
    pos3 = all_str.find(']', pos2)
    scale = np.fromstring(all_str[pos2 + 1:pos3], dtype=np.float32, sep=' ')
    fp.close()
    return mean, scale


class Feature:
    r"""Extract feat from one utterance.
    """

    def __init__(self,
                 fbank_config,
                 feat_type='spec',
                 mvn_file=None,
                 cuda=False):
        r"""

        Args:
            fbank_config (dict):
            feat_type (str):
                raw: do nothing
                fbank: use kaldi.fbank
                spec: Real/Imag
                logpow: log(1+|x|^2)
            mvn_file (str): the path of data file for mean variance normalization
            cuda:
        """
        self.fbank_config = fbank_config
        self.feat_type = feat_type
        self.n_fft = fbank_config['frame_length'] * fbank_config[
            'sample_frequency'] // 1000
        self.hop_length = fbank_config['frame_shift'] * fbank_config[
            'sample_frequency'] // 1000
        self.window = torch.hamming_window(self.n_fft, periodic=False)

        self.mvn = False
        if mvn_file is not None and os.path.exists(mvn_file):
            print(f'loading mvn file: {mvn_file}')
            shift, scale = load_kaldi_feature_transform(mvn_file)
            self.shift = torch.from_numpy(shift)
            self.scale = torch.from_numpy(scale)
            self.mvn = True
        if cuda:
            self.window = self.window.cuda()
            if self.mvn:
                self.shift = self.shift.cuda()
                self.scale = self.scale.cuda()

    def compute(self, utt):
        r"""

        Args:
            utt: in [-32768, 32767] range

        Returns:
             [..., T, F]
        """
        if self.feat_type == 'raw':
            return utt
        elif self.feat_type == 'fbank':
            # have to use local import before modelscope framework supoort lazy loading
            import torchaudio.compliance.kaldi as kaldi
            if len(utt.shape) == 1:
                utt = utt.unsqueeze(0)
            feat = kaldi.fbank(utt, **self.fbank_config)
        elif self.feat_type == 'spec':
            spec = torch.stft(
                utt / 32768,
                self.n_fft,
                self.hop_length,
                self.n_fft,
                self.window,
                center=False,
                return_complex=True)
            feat = torch.cat([spec.real, spec.imag], dim=-2).permute(-1, -2)
        elif self.feat_type == 'logpow':
            spec = torch.stft(
                utt,
                self.n_fft,
                self.hop_length,
                self.n_fft,
                self.window,
                center=False,
                return_complex=True)
            abspow = torch.abs(spec)**2
            feat = torch.log(1 + abspow).permute(-1, -2)
        return feat

    def normalize(self, feat):
        if self.mvn:
            feat = feat + self.shift
            feat = feat * self.scale
        return feat


@PREPROCESSORS.register_module(Fields.audio)
class LinearAECAndFbank:
    SAMPLE_RATE = 16000

    def __init__(self, io_config):
        self.trunc_length = 7200 * self.SAMPLE_RATE
        self.linear_aec_delay = io_config['linear_aec_delay']
        self.feature = Feature(io_config['fbank_config'],
                               io_config['feat_type'], io_config['mvn'])
        self.mitaec = load_library(io_config['mitaec_library'])
        self.mask_on_mic = io_config['mask_on'] == 'nearend_mic'

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ linear filtering the near end mic and far end audio, then extract the feature
        :param data: dict with two keys and correspond audios: "nearend_mic" and "farend_speech"
        :return: dict with two keys and Tensor values: "base" linear filtered audio，and "feature"
        """
        # read files
        nearend_mic, fs = load_wav(data['nearend_mic'])
        assert fs == self.SAMPLE_RATE, f'The sample rate should be {self.SAMPLE_RATE}'
        farend_speech, fs = load_wav(data['farend_speech'])
        assert fs == self.SAMPLE_RATE, f'The sample rate should be {self.SAMPLE_RATE}'
        if 'nearend_speech' in data:
            nearend_speech, fs = load_wav(data['nearend_speech'])
            assert fs == self.SAMPLE_RATE, f'The sample rate should be {self.SAMPLE_RATE}'
        else:
            nearend_speech = np.zeros_like(nearend_mic)

        out_mic, out_ref, out_linear, out_echo = do_linear_aec(
            self.mitaec, nearend_mic, farend_speech)
        # fix 20ms linear aec delay by delaying the target speech
        extra_zeros = np.zeros([int(self.linear_aec_delay * fs)])
        nearend_speech = np.concatenate([extra_zeros, nearend_speech])
        # truncate files to the same length
        flen = min(
            len(out_mic), len(out_ref), len(out_linear), len(out_echo),
            len(nearend_speech))
        fstart = 0
        flen = min(flen, self.trunc_length)
        nearend_mic, out_ref, out_linear, out_echo, nearend_speech = (
            out_mic[fstart:flen], out_ref[fstart:flen],
            out_linear[fstart:flen], out_echo[fstart:flen],
            nearend_speech[fstart:flen])

        # extract features (frames, [mic, linear, ref, aes?])
        feat = torch.FloatTensor()

        nearend_mic = torch.from_numpy(np.float32(nearend_mic))
        fbank_nearend_mic = self.feature.compute(nearend_mic)
        feat = torch.cat([feat, fbank_nearend_mic], dim=1)

        out_linear = torch.from_numpy(np.float32(out_linear))
        fbank_out_linear = self.feature.compute(out_linear)
        feat = torch.cat([feat, fbank_out_linear], dim=1)

        out_echo = torch.from_numpy(np.float32(out_echo))
        fbank_out_echo = self.feature.compute(out_echo)
        feat = torch.cat([feat, fbank_out_echo], dim=1)

        # feature transform
        feat = self.feature.normalize(feat)

        # prepare target
        if nearend_speech is not None:
            nearend_speech = torch.from_numpy(np.float32(nearend_speech))

        if self.mask_on_mic:
            base = nearend_mic
        else:
            base = out_linear
        out_data = {'base': base, 'target': nearend_speech, 'feature': feat}
        return out_data

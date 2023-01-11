# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import scipy.io.wavfile as wav
import torch

from modelscope.fileio import File
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys


class AudioBrainPreprocessor(Preprocessor):
    """A preprocessor takes audio file path and reads it into tensor

    Args:
        takes: the audio file field name
        provides: the tensor field name
        mode: process mode, default 'inference'
    """

    def __init__(self,
                 takes: str,
                 provides: str,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        super(AudioBrainPreprocessor, self).__init__(mode, *args, **kwargs)
        self.takes = takes
        self.provides = provides
        import speechbrain as sb
        self.read_audio = sb.dataio.dataio.read_audio

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = self.read_audio(data[self.takes])
        data[self.provides] = result
        return data


def load_kaldi_feature_transform(filename):
    fp = open(filename, 'r', encoding='utf-8')
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
class LinearAECAndFbank(Preprocessor):
    SAMPLE_RATE = 16000

    def __init__(self, io_config):
        import MinDAEC
        self.trunc_length = 7200 * self.SAMPLE_RATE
        self.linear_aec_delay = io_config['linear_aec_delay']
        self.feature = Feature(io_config['fbank_config'],
                               io_config['feat_type'], io_config['mvn'])
        self.mitaec = MinDAEC.load()
        self.mask_on_mic = io_config['mask_on'] == 'nearend_mic'

    def __call__(self, data: Union[Tuple, Dict[str, Any]]) -> Dict[str, Any]:
        """ Linear filtering the near end mic and far end audio, then extract the feature.

        Args:
            data: Dict with two keys and correspond audios: "nearend_mic" and "farend_speech".

        Returns:
            Dict with two keys and Tensor values: "base" linear filtered audio，and "feature"
        """
        if isinstance(data, tuple):
            nearend_mic, fs = self.load_wav(data[0])
            farend_speech, fs = self.load_wav(data[1])
            nearend_speech = np.zeros_like(nearend_mic)
        else:
            # read files
            nearend_mic, fs = self.load_wav(data['nearend_mic'])
            farend_speech, fs = self.load_wav(data['farend_speech'])
            if 'nearend_speech' in data:
                nearend_speech, fs = self.load_wav(data['nearend_speech'])
            else:
                nearend_speech = np.zeros_like(nearend_mic)

        out_mic, out_ref, out_linear, out_echo = self.mitaec.do_linear_aec(
            nearend_mic, farend_speech)
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

    @staticmethod
    def load_wav(inputs):
        import librosa
        if isinstance(inputs, bytes):
            inputs = io.BytesIO(inputs)
        elif isinstance(inputs, str):
            file_bytes = File.read(inputs)
            inputs = io.BytesIO(file_bytes)
        else:
            raise TypeError(f'Unsupported input type: {type(inputs)}.')
        sample_rate, data = wav.read(inputs)
        if len(data.shape) > 1:
            raise ValueError('modelscope error:The audio must be mono.')
        if sample_rate != LinearAECAndFbank.SAMPLE_RATE:
            data = librosa.resample(data, sample_rate,
                                    LinearAECAndFbank.SAMPLE_RATE)
        return data.astype(np.float32), LinearAECAndFbank.SAMPLE_RATE

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence


def load_cmvn(cmvn_file):
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    # dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append(
                (inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n:]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)


class WavFrontend(torch.nn.Module):

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = 'hamming',
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        snip_edges: bool = True,
        upsacle_samples: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples
        self.cmvn = None if self.cmvn_file is None else load_cmvn(
            self.cmvn_file)

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
        self,
        input: torch.Tensor,
        input_lengths,
        **kwargs,
    ):
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                # print(waveform )
                waveform = waveform * (1 << 15)
                # print(waveform)
            waveform = waveform.unsqueeze(0)
            # print('fbank:',self.upsacle_samples,self.n_mels,self.frame_length,self.frame_shift,self.dither,self.window,self.fs,self.snip_edges)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
            )
            # print("front",mat.shape)
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(
                feats, batch_first=True, padding_value=0.0)
        # print(feats_pad.shape,feats_lens)
        return feats_pad, feats_lens

    def forward_fbank(self, input: torch.Tensor, input_lengths: torch.Tensor):
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
            )

            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(self, input: torch.Tensor,
                         input_lengths: torch.Tensor):
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            mat = input[i, :input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):

    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None
            for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class Encoder:

    def __init__(self, encoder_front_path, encoder_onnx_path):
        self.front = WavFrontend(
            encoder_front_path, lfr_m=7, lfr_n=6, dither=0.0)
        self.asr_session = onnxruntime.InferenceSession(
            encoder_onnx_path,
            provider_options=onnxruntime.get_available_providers())

    def inference(self, wav_path):
        wav = librosa.load(wav_path, sr=16000)[0]
        wav_len = len(wav)
        wav = wav.reshape([1, -1])
        wav = torch.FloatTensor(wav)
        wav_len = torch.IntTensor(np.array([wav_len]))

        feats, feats_len = self.front(wav, wav_len)
        feats = feats.detach().cpu().numpy()
        # print(feats.shape)
        masks = ~make_pad_mask(feats_len)[:, None, :]

        outs = self.asr_session.run(
            ['ys_pad', 'olens'],
            input_feed={
                'xs_pad': feats,
                'masks': masks.cpu().detach().numpy().astype('float32')
            })
        return torch.FloatTensor(outs[0])

    def get_feats(self, wav_path):
        wav = librosa.load(wav_path, sr=16000)[0]
        wav_len = len(wav)
        wav = wav.reshape([1, -1])
        wav = torch.FloatTensor(wav)
        wav_len = torch.IntTensor(np.array([wav_len]))

        feats, feats_len = self.front(wav, wav_len)
        return feats

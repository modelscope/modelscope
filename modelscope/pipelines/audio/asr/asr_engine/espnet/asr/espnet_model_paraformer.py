# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict, List, Optional, Tuple, Union

import torch
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import \
    LabelSmoothingLoss  # noqa: H301
from typeguard import check_argument_types

from ...espnet.nets.pytorch_backend.cif_utils.cif import \
    CIF_Model as cif_predictor

if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class Paraformer(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = '<space>',
        sym_blank: str = '<blank>',
        extract_feats_in_collect_stats: bool = True,
        predictor=None,
        predictor_weight: float = 0.0,
        glat_context_p: float = 0.2,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, 'interctc_use_conditioning'):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size())

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            # from warprnnt_pytorch import RNNTLoss
            from warp_rnnt import rnnt_loss as RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer,
                        report_wer)
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer)

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.glat_context_p = glat_context_p
        self.criterion_pre = torch.nn.L1Loss()
        self.step_cur = 0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]), \
            (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        self.step_cur += 1
        # for data-parallel
        text = text[:, :text_lengths.max()]
        speech = speech[:, :speech_lengths.max(), :]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        loss_pre = None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out,
                                                    encoder_out_lens, text,
                                                    text_lengths)

            # Collect CTC branch stats
            stats['loss_ctc'] = loss_ctc.detach(
            ) if loss_ctc is not None else None
            stats['cer_ctc'] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(intermediate_out,
                                                      encoder_out_lens, text,
                                                      text_lengths)
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats['loss_interctc_layer{}'.format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None)
                stats['cer_interctc_layer{}'.format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (1 - self.interctc_weight
                        ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats['loss_transducer'] = (
                loss_transducer.detach()
                if loss_transducer is not None else None)
            stats['cer_transducer'] = cer_transducer
            stats['wer_transducer'] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:

                loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths)

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (
                    1 - self.ctc_weight
                ) * loss_att + loss_pre * self.predictor_weight

            # Collect Attn branch stats
            stats['loss_att'] = loss_att.detach(
            ) if loss_att is not None else None
            stats['acc'] = acc_att
            stats['cer'] = cer_att
            stats['wer'] = wer_att
            stats['loss_pre'] = loss_pre.detach().cpu(
            ) if loss_pre is not None else None

        # Collect total loss stats
        # TODO(wjm): needed to be checked
        # TODO(wjm): same problem: https://github.com/espnet/espnet/issues/4136
        # FIXME(wjm): for logger error when accum_grad > 1
        # stats["loss"] = loss.detach()
        stats['loss'] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size),
                                               loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                'Generating dummy stats for feats and feats_lengths, '
                'because encoder_conf.extract_feats_in_collect_stats is '
                f'{self.extract_feats_in_collect_stats}')
            feats, feats_lengths = speech, speech_lengths
        return {'feats': feats, 'feats_lengths': feats_lengths}

    def encode(
            self, speech: torch.Tensor,
            speech_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc)
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (~make_pad_mask(
            encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
                encoder_out.device)
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length

    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens,
                                   sematic_embeds, ys_pad_lens):

        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens,
                                      sematic_embeds, ys_pad_lens)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def _extract_feats(
            self, speech: torch.Tensor,
            speech_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, :speech_lengths.max()]
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad,
                                      ys_in_lens)  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction='none',
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        encoder_out_mask = (~make_pad_mask(
            encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
                encoder_out.device)
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id)

        # 0. sampler
        decoder_out_1st = None
        if self.glat_context_p > 0.0:
            if self.step_cur < 2:
                logging.info(
                    'enable sampler in paraformer, glat_context_p: {}'.format(
                        self.glat_context_p))
            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                pre_acoustic_embeds)
        else:
            if self.step_cur < 2:
                logging.info(
                    'disable sampler in paraformer, glat_context_p: {}'.format(
                        self.glat_context_p))
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                    sematic_embeds, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(
            ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(),
                                                     ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                pre_acoustic_embeds):

        tgt_mask = (~make_pad_mask(ys_pad_lens,
                                   maxlen=ys_pad_lens.max())[:, :, None]).to(
                                       ys_pad.device)
        ys_pad *= tgt_mask[:, :, 0]
        ys_pad_embed = self.decoder.embed(ys_pad)
        with torch.no_grad():
            decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                        pre_acoustic_embeds, ys_pad_lens)
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float())
                              * self.glat_context_p).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                        value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(
                pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(
            ~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
                input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def _calc_att_loss_ar(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad,
                                      ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(),
                                                     ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(
                ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1))

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
            reduction='sum',
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target)

        return loss_transducer, cer_transducer, wer_transducer


class ParaformerBertEmbed(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = '<space>',
        sym_blank: str = '<blank>',
        extract_feats_in_collect_stats: bool = True,
        predictor: cif_predictor = None,
        predictor_weight: float = 0.0,
        glat_context_p: float = 0.2,
        embed_dims: int = 768,
        embeds_loss_weight: float = 0.0,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, 'interctc_use_conditioning'):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size())

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            # from warprnnt_pytorch import RNNTLoss
            from warp_rnnt import rnnt_loss as RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer,
                        report_wer)
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer)

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.glat_context_p = glat_context_p
        self.criterion_pre = torch.nn.L1Loss()
        self.step_cur = 0
        self.pro_nn = torch.nn.Linear(encoder.output_size(), embed_dims)
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.embeds_loss_weight = embeds_loss_weight
        self.length_normalized_loss = length_normalized_loss

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        embed: torch.Tensor = None,
        embed_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]), \
            (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        self.step_cur += 1
        # for data-parallel
        text = text[:, :text_lengths.max()]
        speech = speech[:, :speech_lengths.max(), :]
        if embed is not None:
            embed = embed[:, :embed_lengths.max(), :]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        loss_pre = None
        cos_loss = None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out,
                                                    encoder_out_lens, text,
                                                    text_lengths)

            # Collect CTC branch stats
            stats['loss_ctc'] = loss_ctc.detach(
            ) if loss_ctc is not None else None
            stats['cer_ctc'] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0

        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(intermediate_out,
                                                      encoder_out_lens, text,
                                                      text_lengths)
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats['loss_interctc_layer{}'.format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None)
                stats['cer_interctc_layer{}'.format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (1 - self.interctc_weight
                        ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats['loss_transducer'] = (
                loss_transducer.detach()
                if loss_transducer is not None else None)
            stats['cer_transducer'] = cer_transducer
            stats['wer_transducer'] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:

                if embed is None or self.embeds_loss_weight <= 0.0:
                    loss_ret = self._calc_att_loss(encoder_out,
                                                   encoder_out_lens, text,
                                                   text_lengths)
                    loss_att, acc_att, cer_att, wer_att, loss_pre = loss_ret[
                        0], loss_ret[1], loss_ret[2], loss_ret[3], loss_ret[4]
                else:
                    loss_ret = self._calc_att_loss_embed(
                        encoder_out, encoder_out_lens, text, text_lengths,
                        embed, embed_lengths)
                    loss_att, acc_att, cer_att, wer_att, loss_pre = loss_ret[
                        0], loss_ret[1], loss_ret[2], loss_ret[3], loss_ret[4]
                    embeds_outputs = None
                    if len(loss_ret) > 5:
                        embeds_outputs = loss_ret[5]
                    if embeds_outputs is not None:
                        cos_loss = self._calc_embed_loss(
                            text, text_lengths, embed, embed_lengths,
                            embeds_outputs)
            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            elif self.embeds_loss_weight > 0.0:
                loss = self.ctc_weight * loss_ctc + (
                    1 - self.ctc_weight
                ) * loss_att + loss_pre * self.predictor_weight + cos_loss * self.embeds_loss_weight
            else:
                loss = self.ctc_weight * loss_ctc + (
                    1 - self.ctc_weight
                ) * loss_att + loss_pre * self.predictor_weight

            # Collect Attn branch stats
            stats['loss_att'] = loss_att.detach(
            ) if loss_att is not None else None
            stats['acc'] = acc_att
            stats['cer'] = cer_att
            stats['wer'] = wer_att
            stats['loss_pre'] = loss_pre.detach().cpu(
            ) if loss_pre is not None else None
            stats['cos_loss'] = cos_loss.detach().cpu(
            ) if cos_loss is not None else None

        # Collect total loss stats
        # TODO(wjm): needed to be checked
        # TODO(wjm): same problem: https://github.com/espnet/espnet/issues/4136
        # FIXME(wjm): for logger error when accum_grad > 1
        # stats["loss"] = loss.detach()
        stats['loss'] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size),
                                               loss.device)
        return loss, stats, weight

    def _calc_embed_loss(
        self,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        embed: torch.Tensor = None,
        embed_lengths: torch.Tensor = None,
        embeds_outputs: torch.Tensor = None,
    ):
        embeds_outputs = self.pro_nn(embeds_outputs)
        tgt_mask = (~make_pad_mask(ys_pad_lens,
                                   maxlen=ys_pad_lens.max())[:, :, None]).to(
                                       ys_pad.device)
        embeds_outputs *= tgt_mask  # b x l x d
        embed *= tgt_mask  # b x l x d
        cos_loss = 1.0 - self.cos(embeds_outputs, embed)
        cos_loss *= tgt_mask.squeeze(2)
        if self.length_normalized_loss:
            token_num_total = torch.sum(tgt_mask)
        else:
            token_num_total = tgt_mask.size()[0]
        cos_loss_total = torch.sum(cos_loss)
        cos_loss = cos_loss_total / token_num_total
        return cos_loss

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                'Generating dummy stats for feats and feats_lengths, '
                'because encoder_conf.extract_feats_in_collect_stats is '
                f'{self.extract_feats_in_collect_stats}')
            feats, feats_lengths = speech, speech_lengths
        return {'feats': feats, 'feats_lengths': feats_lengths}

    def encode(
            self, speech: torch.Tensor,
            speech_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc)
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (~make_pad_mask(
            encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
                encoder_out.device)
        # logging.info(
        # 	"encoder_out_mask size: {}".format(encoder_out_mask.size()))
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length

    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens,
                                   sematic_embeds, ys_pad_lens):

        decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                    sematic_embeds, ys_pad_lens)
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def _extract_feats(
            self, speech: torch.Tensor,
            speech_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, :speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad,
                                      ys_in_lens)  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction='none',
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        encoder_out_mask = (~make_pad_mask(
            encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
                encoder_out.device)
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id)

        # 0. sampler
        decoder_out_1st = None
        if self.glat_context_p > 0.0:
            if self.step_cur < 2:
                logging.info(
                    'enable sampler in paraformer, glat_context_p: {}'.format(
                        self.glat_context_p))
            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                pre_acoustic_embeds)
        else:
            if self.step_cur < 2:
                logging.info(
                    'disable sampler in paraformer, glat_context_p: {}'.format(
                        self.glat_context_p))
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                    sematic_embeds, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(
            ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(),
                                                     ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre

    def _calc_att_loss_embed(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        embed: torch.Tensor = None,
        embed_lengths: torch.Tensor = None,
    ):
        encoder_out_mask = (~make_pad_mask(
            encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
                encoder_out.device)
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id)

        # 0. sampler
        decoder_out_1st = None
        if self.glat_context_p > 0.0:
            if self.step_cur < 2:
                logging.info(
                    'enable sampler in paraformer, glat_context_p: {}'.format(
                        self.glat_context_p))
            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                pre_acoustic_embeds)
        else:
            if self.step_cur < 2:
                logging.info(
                    'disable sampler in paraformer, glat_context_p: {}'.format(
                        self.glat_context_p))
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                    sematic_embeds, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        if len(decoder_outs) > 2:
            embeds_outputs = decoder_outs[2]
        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(
            ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(),
                                                     ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre, embeds_outputs

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                pre_acoustic_embeds):

        tgt_mask = (~make_pad_mask(ys_pad_lens,
                                   maxlen=ys_pad_lens.max())[:, :, None]).to(
                                       ys_pad.device)
        ys_pad *= tgt_mask[:, :, 0]
        ys_pad_embed = self.decoder.embed(ys_pad)
        with torch.no_grad():
            decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                        pre_acoustic_embeds, ys_pad_lens)
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float())
                              * self.glat_context_p).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                        value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(
                pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(
            ~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
                input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def _calc_att_loss_ar(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad,
                                      ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(),
                                                     ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(
                ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1))

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
            reduction='sum',
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target)

        return loss_transducer, cer_transducer, wer_transducer

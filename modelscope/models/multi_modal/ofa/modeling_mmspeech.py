# Copyright 2022 OFA-Sys Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OFA-MMSpeech model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
from fairseq.modules import LayerNorm, SamePad, TransposeLast
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
from packaging import version
from torch import nn
from torch.nn import functional as F
from transformers.file_utils import (ModelOutput, add_code_sample_docstrings,
                                     add_start_docstrings,
                                     add_start_docstrings_to_model_forward)
from transformers.utils import logging

from .configuration_mmspeech import MMSpeechConfig
from .generate import utils
from .modeling_ofa import (Embedding, OFADecoder, OFAModel, OFAPreTrainedModel,
                           _expand_mask)

logger = logging.get_logger()

_CHECKPOINT_FOR_DOC = 'mmspeech-base'
_CONFIG_FOR_DOC = 'MMSpeechConfig'
_TOKENIZER_FOR_DOC = 'OFATokenizer'
TORCH_VERSION = version.parse(torch.__version__)
TORCH_MESH_GRID_WARNING_VERSION = version.parse('1.9.1')

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

OFA_PRETRAINED_MODEL_ARCHIVE_LIST = ['mmspeech-base', 'mmspeech-large']

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):

        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


class MMSpeechPreTrainedModel(OFAPreTrainedModel):
    r"""
    Base class OFA
    """

    config_class = MMSpeechConfig

    def _set_gradient_checkpointing(self, module, value=False):
        r"""
        Turn on the switch of gradient checkpointing.
        """
        if isinstance(module, (OFADecoder, MMSpeechEncoder)):
            module.gradient_checkpointing = value


@dataclass
class MMSpeechEncoderOutput(ModelOutput):
    r"""
    Base class for OFA's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`):
            Sequence of hidden-states at the output of the last layer of the model.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):

            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(bsz, seq_len, hidden)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):

            Tuple of `torch.FloatTensor` (one for each layer) of shape `(bsz, num_heads, seq_len, seq_len)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        position_embedding (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`):
            postional embeddings of the inputs.
    """

    phone_distribution: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    padding_mask: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    position_embedding: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.Tensor] = None


@dataclass
class MMSpeechModelOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*,
            returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax,
            used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer,
            after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`,
            *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_padding_mask: Optional[torch.Tensor] = None
    phone_distribution: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None


MMSPEECH_START_DOCSTRING = r"""
    This model inherits from [`OFAModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`~MMSpeechConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MMSPEECH_GENERATION_EXAMPLE = r"""
    Image captioning example:

    ```python
    >>> import soundfile as sf
    >>> import torchaudio
    >>> import torchaudio.compliance.kaldi as ta_kaldi
    >>> wav, sr = sf.read(data[self.column_map['wav']])
    >>> wav = torchaudio.sox_effects.apply_effects_tensor(
    >>>         wav, sr,
    >>>         [['speed', '1.0'], ['rate', '16000'], ['gain', '-n'], ['channels', '1']]))
    >>> wav = wav * (2**15)
    >>> wav = torch.from_numpy(wav.numpy())
    >>> fbank = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
    >>> fbank_mask = torch.tensor([True])
    >>> model = MMSpeechModel.from_pretrained(ckpt_dir)
    >>> tokenizer = OFATokenizerZH.from_pretrained(ckpt_dir)

    >>> gen = model.generate(fbank=fbank, fbank_mask=fbank_mask, num_beams=4)
    >>> print(tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""

MMSPEECH_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
            indices of input sequence tokens in the vocabular, and padding will be ignored by default;

            indices can be obtained using [`~OFATokenizer`].

        patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
            the resized image, which are transformed by the default operations.
        patch_images_2 (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
            the second (if it exists) image.
        patch_masks (`torch.BoolTensor`): the patches to be masked.
        token_embeddings (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): token embeddings.
        sample_patch_num (`int`): the number of patches to sample.
        fbank (`torch.Tensor`): fbank feature of audio.
        fbank_length (`torch.Tensor`): fbank length of audio.
        fbank_masks (`torch.BoolTensor`): whether to have fbank feature.
        phone_items (`torch.Tensor`): phoneme sequence.
        phone_masks (`torch.BoolTensor`): whether to have phoneme feature.
        features_only (`torch.BoolTensor`): whether to return encoder features only.
        mask (`torch.BoolTensor`): whether to mask fbank feature.
        mask_prob (`torch.Tensor`): the prob of mask fbank feature.
        layer (`int`): the number of layer to cache hidden state.
        decoder_input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`): indices of the sequence in the vocabulary.
        code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
        attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): attention mask for decoding.
        encoder_outputs (`OFAEncoderOutput`):
            encoder outputs with hidden states, positional embeddings, and padding masks.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of
            shape `(bsz, num_heads, src_len, head_size)`.
        use_cache (`bool`): whether to use cache for faster inference.
        output_attentions (`bool`): whether to output attention weights.
        output_hidden_states (`bool`): whether to output hidden states.
        return_dict (`bool`): unused. Keep it for generation only.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
"""


class Conv2dSubsampling4(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(2):
            out = ((out.float() - 1) // 2 + 1).floor().long()
        return out

    def forward(self, x: torch.Tensor,
                x_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        return x, self.get_out_seq_lens_tensor(x_length)


class TransformerEncoder(nn.Module):

    def build_encoder_layer(self, args: MMSpeechConfig):
        layer = TransformerSentenceEncoderLayer(
            embedding_dim=self.embedding_dim,
            ffn_embedding_dim=args.encoder_ffn_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=self.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            activation_fn=args.activation_function,
            layer_norm_first=args.encoder_normalize_before,
        )
        return layer

    def __init__(self, args: MMSpeechConfig):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.d_model
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = args.encoder_pos_conv_depth
        if pos_conv_depth > 1:
            num_layers = args.encoder_pos_conv_depth
            k = max(3, args.encoder_conv_pos // num_layers)

            def make_conv_block(e, k, g, la):
                return nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv1d(
                            e,
                            e,
                            kernel_size=k,
                            padding=k // 2,
                            groups=g,
                        ),
                        SamePad(k),
                        TransposeLast(),
                        LayerNorm(e, elementwise_affine=False),
                        TransposeLast(),
                        nn.GELU(),
                    ) for _ in range(la)
                ])

            self.pos_conv = make_conv_block(self.embedding_dim, k,
                                            args.encoder_conv_pos_groups,
                                            num_layers)
            self.phone_pos_conv = make_conv_block(self.embedding_dim, k,
                                                  args.encoder_conv_pos_groups,
                                                  num_layers)

        else:

            def make_conv_pos(e, k, g):
                pos_conv = nn.Conv1d(
                    e,
                    e,
                    kernel_size=k,
                    padding=k // 2,
                    groups=g,
                )
                dropout = 0
                std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
                nn.init.normal_(pos_conv.weight, mean=0, std=std)
                nn.init.constant_(pos_conv.bias, 0)

                pos_conv = nn.utils.weight_norm(pos_conv, name='weight', dim=2)
                pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

                return pos_conv

            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.encoder_conv_pos,
                args.encoder_conv_pos_groups,
            )
            self.phone_pos_conv = make_conv_pos(
                self.embedding_dim,
                args.encoder_conv_pos,
                args.encoder_conv_pos_groups,
            )

        self.layers = nn.ModuleList([
            self.build_encoder_layer(args) for _ in range(args.encoder_layers)
        ])
        self.layer_norm_first = args.encoder_normalize_before

        self.layer_norm = LayerNorm(self.embedding_dim)
        self.phone_layer_norm = LayerNorm(self.embedding_dim)

        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self,
                x,
                padding_mask=None,
                phone_x=None,
                phone_padding_mask=None,
                layer=None,
                context_layer=None):
        x, layer_results, x_conv, pre_padding_mask = self.extract_features(
            x,
            padding_mask,
            phone_x,
            phone_padding_mask,
            layer,
            context_layer=context_layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results, x_conv, pre_padding_mask

    def extract_features(
        self,
        x,
        padding_mask=None,
        phone_x=None,
        phone_padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        context_layer=None,
    ):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        if phone_x is not None:
            if phone_padding_mask is not None:
                phone_x = index_put(phone_x, phone_padding_mask, 0)

            phone_x_conv = self.phone_pos_conv(phone_x.transpose(1, 2))
            phone_x_conv = phone_x_conv.transpose(1, 2)
            phone_x = phone_x + phone_x_conv

            if not self.layer_norm_first:
                # to fix
                phone_x = self.layer_norm(phone_x)

        pre_padding_mask = padding_mask.clone()

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):

            if i < context_layer and (~padding_mask).any() is False:
                continue

            if i == context_layer and phone_x is not None and phone_x_conv is not None:
                x = x.transpose(0, 1)
                x = torch.cat([x, phone_x], dim=1)
                padding_mask = torch.cat([padding_mask, phone_padding_mask],
                                         dim=1)
                pre_padding_mask = padding_mask.clone()
                x_conv = torch.cat([x_conv, phone_x_conv], dim=1)
                x = x.transpose(0, 1)

            dropout_probability = np.random.random(
            ) if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False)
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results, x_conv, pre_padding_mask

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.encoder_max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class MMSpeechEncoder(MMSpeechPreTrainedModel):

    def __init__(self,
                 cfg: MMSpeechConfig,
                 embed_tokens: Optional[nn.Embedding] = None):

        super().__init__(cfg)

        self.cfg = cfg

        self.embed = cfg.d_model

        # fbank encoder
        self.subsample = Conv2dSubsampling4(80 * 1, cfg.d_model)
        self.post_subsample_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # phone and text encoder
        self.padding_idx = embed_tokens.padding_idx
        self.phone_padding_idx = self.padding_idx
        self.phone_item_embedding = Embedding(cfg.phone_vocab_size, self.embed,
                                              self.phone_padding_idx)

        # mask
        self.mask_prob = cfg.audio_mask_prob
        self.mask_selection = cfg.audio_mask_selection
        self.mask_other = cfg.audio_mask_other
        self.mask_length = cfg.audio_mask_length
        self.no_mask_overlap = cfg.audio_no_mask_overlap
        self.mask_min_space = cfg.audio_mask_min_space

        self.mask_channel_prob = cfg.audio_mask_channel_prob
        self.mask_channel_before = cfg.audio_mask_channel_before
        self.mask_channel_selection = cfg.audio_mask_channel_selection
        self.mask_channel_other = cfg.audio_mask_channel_other
        self.mask_channel_length = cfg.audio_mask_channel_length
        self.no_mask_channel_overlap = cfg.audio_no_mask_channel_overlap
        self.mask_channel_min_space = cfg.audio_mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.encoder_dropout_input)
        self.dropout_features = nn.Dropout(cfg.encoder_dropout_features)

        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.d_model).uniform_())

        self.encoder = TransformerEncoder(cfg)

        self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0

    def get_input_embeddings(self):
        r"""
        Get the embedding weight.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        r"""
        Set the weight of embedding with the given tensor.
        """
        self.embed_tokens = value

    def apply_mask(self,
                   x,
                   padding_mask,
                   mask_indices=None,
                   mask_channel_indices=None,
                   mask_prob=None):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).to(
                    x.device).unsqueeze(1).expand(-1, T, -1))
            x[mask_channel_indices] = 0

        if self.mask_prob > 0 or mask_prob is not None:
            if mask_indices is None:
                if mask_prob is None:
                    mask_prob = self.mask_prob
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices).to(
                        x.device).unsqueeze(1).expand(-1, T, -1))
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self,
                                         input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths,
                                             conv_cfg_list[i][1],
                                             conv_cfg_list[i][2])

        return input_lengths.to(torch.long)

    def forward(self,
                fbank: Optional[torch.Tensor] = None,
                fbank_length: Optional[torch.Tensor] = None,
                fbank_masks: Optional[torch.Tensor] = None,
                phone_items: Optional[torch.Tensor] = None,
                phone_masks: Optional[torch.Tensor] = None,
                features_only: Optional[torch.Tensor] = True,
                mask: Optional[torch.Tensor] = False,
                mask_prob: Optional[torch.Tensor] = None,
                layer=None,
                output_hidden_states=False):

        features, fbank_feature_length = self.subsample(fbank, fbank_length)

        if self.post_subsample_proj is not None:
            features = self.post_subsample_proj(features)

        padding_mask = (
            torch.BoolTensor(features.shape[:2]).fill_(False)
            # if self.pad_audio else None
        ).to(features.device)
        for i, l in enumerate(fbank_feature_length):
            diff = l - padding_mask.shape[-1]
            if diff < 0:
                padding_mask[i, diff:] = True

        pre_encoder_features = features.clone()
        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask, mask_prob=mask_prob)
        else:
            x = features
            mask_indices = None

        padding_mask[~fbank_masks] = True

        phone_x = None
        phone_padding_mask = None
        if phone_items is not None:
            phone_x = self.phone_item_embedding(phone_items)
            phone_padding_mask = phone_items.eq(self.phone_padding_idx)
            phone_padding_mask[~phone_masks] = True
            if mask_indices is not None:
                phone_mask_indices = phone_padding_mask.new_zeros(
                    phone_padding_mask.size()).bool()
                mask_indices = torch.cat([mask_indices, phone_mask_indices],
                                         dim=1)

        pre_padding_mask = padding_mask.clone()
        x, layer_results, pos_embed, padding_mask = self.encoder(
            x,
            padding_mask=padding_mask,
            phone_x=phone_x,
            phone_padding_mask=phone_padding_mask,
            layer=layer,
            context_layer=6)

        emb_weight = self.phone_item_embedding.weight[
            3:self.cfg.phone_dict_size, :]
        if features_only is False:  # no gradient for embedding here
            emb_weight = emb_weight.detach()

        phone_distribution = F.linear(x, emb_weight, None)

        if features_only:
            return MMSpeechEncoderOutput(
                phone_distribution=phone_distribution.transpose(0, 1),
                last_hidden_state=x,
                padding_mask=padding_mask,
                position_embedding=pos_embed)

        result = {
            'losses': {},
        }

        with torch.no_grad():
            self.encoder.eval()
            y, y_layer_results, _, _ = self.encoder.extract_features(
                pre_encoder_features,
                padding_mask=pre_padding_mask,
                phone_x=phone_x,
                phone_padding_mask=phone_padding_mask,
                min_layer=
                0,  # self.cfg.encoder_layers - self.average_top_k_layers,
                context_layer=6)
            y = {
                'x': y,
                'padding_mask': padding_mask,
                'layer_results': y_layer_results,
            }

            emb_weight = self.phone_item_embedding.weight[
                3:self.cfg.phone_dict_size, :]

            y = F.linear(y['x'], emb_weight, None)
            y = y[mask_indices]
            self.encoder.train()

        y_student = phone_distribution[mask_indices]

        def _kl_loss(p, q):
            loss = F.kl_div(
                utils.log_softmax(p, dim=-1),
                utils.softmax(q, dim=-1),
                reduction='sum')
            return loss

        y = y
        kl_loss = _kl_loss(y_student.float(), y.float())

        with torch.no_grad():
            result['target_var'] = self.compute_var(y)
            result['pred_var'] = self.compute_var(y_student.float())

        if self.num_updates > 5000 and result[
                'target_var'] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result[
                'pred_var'] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        return MMSpeechEncoderOutput(
            phone_distribution=phone_distribution.transpose(0, 1),
            last_hidden_state=x,
            padding_mask=padding_mask,
            position_embedding=pos_embed,
            kl_loss=kl_loss)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        # if encoder_out["last_hidden_state"] is None:
        if 'last_hidden_state' not in encoder_out:
            new_encoder_out = None
        else:
            new_encoder_out = encoder_out['last_hidden_state'].index_select(
                0, new_order)
        # if encoder_out["padding_mask"] is None:
        if 'padding_mask' not in encoder_out:
            new_encoder_padding_mask = None
        else:
            new_encoder_padding_mask = encoder_out[
                'padding_mask'].index_select(0, new_order)

        # if encoder_out["position_embedding"] is None:
        if 'position_embedding' not in encoder_out:
            new_position_embeddings = None
        else:
            new_position_embeddings = encoder_out[
                'position_embedding'].index_select(0, new_order)

        if 'hidden_states' not in encoder_out:
            new_encoer_states = None
        else:
            encoder_states = encoder_out['hidden_states']
            new_encoer_states = ()
            if len(encoder_states) > 0:
                for idx, state in enumerate(encoder_states):
                    new_encoer_states += (state.index_select(0, new_order), )

        if 'attentions' not in encoder_out:
            attentions = None
        else:
            attentions = encoder_out['attentions']

        new_kl_loss = None
        if 'kl_loss' in encoder_out:
            new_kl_loss = encoder_out['kl_loss']

        if len(encoder_out['phone_distribution']) == 0:
            new_phone_distribution = None
        else:
            new_phone_distribution = encoder_out[
                'phone_distribution'].index_select(1, new_order)

        return MMSpeechEncoderOutput(
            phone_distribution=new_phone_distribution,
            last_hidden_state=new_encoder_out,  # B x T x C
            padding_mask=new_encoder_padding_mask,  # B x T
            hidden_states=new_encoer_states,  # List[T x B x C]
            attentions=attentions,
            position_embedding=new_position_embeddings,  # B x T x C
            kl_loss=new_kl_loss)

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()


@add_start_docstrings(
    'The bare OFA Model outputting raw hidden-states without any specific head on top.',
    MMSPEECH_START_DOCSTRING,
)
class MMSpeechModel(OFAModel):
    r"""
    The OFA model built with an encoder and a decoder only, without any classification head.

    Args:
        config (MMSpeechConfig): OFA configuration.
    """

    config_class = MMSpeechConfig

    def __init__(self, config: MMSpeechConfig, **kwargs):
        super().__init__(config)
        self.disable_entangle = getattr(kwargs, 'disable_entangle', False)

        self.padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        shared = nn.Embedding(vocab_size, config.d_model, self.padding_idx)

        self.encoder = MMSpeechEncoder(config, shared)
        self.decoder = OFADecoder(config, shared)
        self.use_ofasys = config.use_ofasys

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MMSPEECH_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MMSpeechModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def get_encoder_normalized_probs(self, net_output, log_probs, **kwargs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output['phone_distribution']
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self,
                input_ids=None,
                patch_images=None,
                patch_images_2=None,
                patch_masks=None,
                token_embeddings=None,
                sample_patch_num=None,
                fbank=None,
                fbank_length=None,
                fbank_masks=None,
                phone_items=None,
                phone_masks=None,
                features_only=True,
                mask=False,
                mask_prob=None,
                layer=None,
                decoder_input_ids=None,
                code_masks=None,
                attention_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
                indices of input sequence tokens in the vocabular, and padding will be ignored by default;

                indices can be obtained using [`~OFATokenizer`].

            patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the resized image, which are transformed by the default operations.
            patch_images_2 (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the second (if it exists) image.
            patch_masks (`torch.BoolTensor`): the patches to be masked.
            token_embeddings (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): token embeddings.
            sample_patch_num (`int`): the number of patches to sample.
            decoder_input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`): indices of the sequence in the vocabulary.
            code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
            attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): attention mask for decoding.
            encoder_outputs (`OFAEncoderOutput`):
                encoder outputs with hidden states, positional embeddings, and padding masks.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of
                shape `(bsz, num_heads, src_len, head_size)`.
            use_cache (`bool`): whether to use cache for faster inference.
            output_attentions (`bool`): whether to output attention weights.
            output_hidden_states (`bool`): whether to output hidden states.
            return_dict (`bool`): unused. Keep it for generation only.

        Returns:
            OFASpeechOutput:
                last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`): the last decoder hidden states.
                past_key_values (`tuple(tuple(torch.FloatTensor)): past keys and values for faster inference.
                decoder_hidden_states (`tuple(torch.FloatTensor)`): the decoder hidden states of all layers.
                decoder_attentions (`tuple(torch.FloatTensor)): the decoder self attention weights of all layers.
                cross_attentions (`tuple(torch.FloatTensor)): cross attention weights of all layers.
                encoder_last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`):
                    the encoder last hidden state.
                encoder_hidden_states (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`):
                    the encoder states of all layers including the embeddings.
                encoder_attentions (`torch.FloatTensor` of shape `(bsz, num_heads, seq_len, seq_len)`):
                    the encoder attention weights of all layers.
        """ # noqa

        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                fbank=fbank,
                fbank_length=fbank_length,
                fbank_masks=fbank_masks,
                phone_items=phone_items,
                phone_masks=phone_masks,
                features_only=features_only,
                mask=mask,
                mask_prob=mask_prob,
                layer=layer)

        if decoder_input_ids.eq(self.config.pad_token_id).any():
            attention_mask = decoder_input_ids.eq(self.padding_idx)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = _expand_mask(encoder_outputs.padding_mask,
                                              encoder_hidden_states.dtype,
                                              decoder_input_ids.shape[-1])
        src_pos_embed = encoder_outputs.position_embedding

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            code_masks=code_masks,
            src_pos_embed=src_pos_embed,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return MMSpeechModelOutput(
            logits=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_padding_mask=encoder_outputs.padding_mask,
            phone_distribution=encoder_outputs.phone_distribution,
            kl_loss=encoder_outputs.kl_loss)

    def _set_gradient_checkpointing(self, module, value=False):
        r"""
        Turn on the switch of gradient checkpointing.
        """
        if isinstance(module, (OFADecoder, MMSpeechEncoder)):
            module.gradient_checkpointing = value

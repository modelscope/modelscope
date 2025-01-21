# Part of the implementation is borrowed and modified from FAIRSEQ,
# publicly available at https://github.com/facebookresearch/fairseq
# Copyright 2022-2023 The Alibaba MT Team Authors. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (FairseqEncoder, FairseqEncoderDecoderModel,
                            FairseqIncrementalDecoder, register_model,
                            register_model_architecture)
from fairseq.modules import (AdaptiveSoftmax, BaseLayer, FairseqDropout,
                             LayerDropModuleList, LayerNorm,
                             PositionalEmbedding,
                             SinusoidalPositionalEmbedding,
                             TransformerDecoderLayer, TransformerEncoderLayer)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class CanmtModel(FairseqEncoderDecoderModel):
    """

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The CanmtModel provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder, second_decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.encoder = encoder
        self.decoder = decoder
        self.second_decoder = second_decoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use')
        parser.add_argument(
            '--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument(
            '--attention-dropout',
            type=float,
            metavar='D',
            help='dropout probability for attention weights')
        parser.add_argument(
            '--activation-dropout',
            '--relu-dropout',
            type=float,
            metavar='D',
            help='dropout probability after activation in FFN.')
        parser.add_argument(
            '--encoder-embed-path',
            type=str,
            metavar='STR',
            help='path to pre-trained encoder embedding')
        parser.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension')
        parser.add_argument(
            '--encoder-ffn-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension for FFN')
        parser.add_argument(
            '--encoder-layers',
            type=int,
            metavar='N',
            help='num encoder layers')
        parser.add_argument(
            '--encoder-attention-heads',
            type=int,
            metavar='N',
            help='num encoder attention heads')
        parser.add_argument(
            '--encoder-normalize-before',
            action='store_true',
            help='apply layernorm before each encoder block')
        parser.add_argument(
            '--encoder-learned-pos',
            action='store_true',
            help='use learned positional embeddings in the encoder')
        parser.add_argument(
            '--decoder-embed-path',
            type=str,
            metavar='STR',
            help='path to pre-trained decoder embedding')
        parser.add_argument(
            '--decoder-embed-dim',
            type=int,
            metavar='N',
            help='decoder embedding dimension')
        parser.add_argument(
            '--decoder-ffn-embed-dim',
            type=int,
            metavar='N',
            help='decoder embedding dimension for FFN')
        parser.add_argument(
            '--decoder-layers',
            type=int,
            metavar='N',
            help='num decoder layers')
        parser.add_argument(
            '--decoder-attention-heads',
            type=int,
            metavar='N',
            help='num decoder attention heads')
        parser.add_argument(
            '--decoder-learned-pos',
            action='store_true',
            help='use learned positional embeddings in the decoder')
        parser.add_argument(
            '--decoder-normalize-before',
            action='store_true',
            help='apply layernorm before each decoder block')
        parser.add_argument(
            '--decoder-output-dim',
            type=int,
            metavar='N',
            help='decoder output dimension (extra linear layer '
            'if different from decoder embed dim')
        parser.add_argument(
            '--share-decoder-input-output-embed',
            action='store_true',
            help='share decoder input and output embeddings')
        parser.add_argument(
            '--share-all-embeddings',
            action='store_true',
            help='share encoder, decoder and output embeddings'
            ' (requires shared dictionary and embed dim)')
        parser.add_argument(
            '--no-token-positional-embeddings',
            default=False,
            action='store_true',
            help=
            'if set, disables positional embeddings (outside self attention)')
        parser.add_argument(
            '--adaptive-softmax-cutoff',
            metavar='EXPR',
            help='comma separated list of adaptive softmax cutoff points. '
            'Must be used with adaptive_loss criterion'),
        parser.add_argument(
            '--adaptive-softmax-dropout',
            type=float,
            metavar='D',
            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument(
            '--layernorm-embedding',
            action='store_true',
            help='add layernorm to embedding')
        parser.add_argument(
            '--no-scale-embedding',
            action='store_true',
            help='if True, dont scale embeddings')
        parser.add_argument(
            '--checkpoint-activations',
            action='store_true',
            help='checkpoint activations at each layer, which saves GPU '
            'memory usage at the cost of some additional compute')
        parser.add_argument(
            '--offload-activations',
            action='store_true',
            help='checkpoint activations at each layer, then save to gpu.'
            'Sets --checkpoint-activations.')
        parser.add_argument(
            '--no-cross-attention',
            default=False,
            action='store_true',
            help='do not perform cross-attention')
        parser.add_argument(
            '--cross-self-attention',
            default=False,
            action='store_true',
            help='perform cross+self-attention')
        parser.add_argument(
            '--encoder-layerdrop',
            type=float,
            metavar='D',
            default=0,
            help='LayerDrop probability for encoder')
        parser.add_argument(
            '--decoder-layerdrop',
            type=float,
            metavar='D',
            default=0,
            help='LayerDrop probability for decoder')
        parser.add_argument(
            '--encoder-layers-to-keep',
            default=None,
            help='which layers to *keep* when pruning as a comma-separated list'
        )
        parser.add_argument(
            '--decoder-layers-to-keep',
            default=None,
            help='which layers to *keep* when pruning as a comma-separated list'
        )
        parser.add_argument(
            '--quant-noise-pq',
            type=float,
            metavar='D',
            default=0,
            help='iterative PQ quantization noise at training time')
        parser.add_argument(
            '--quant-noise-pq-block-size',
            type=int,
            metavar='D',
            default=8,
            help='block size of quantization noise at training time')
        parser.add_argument(
            '--quant-noise-scalar',
            type=float,
            metavar='D',
            default=0,
            help=
            'scalar quantization noise and scalar quantization at training time'
        )
        parser.add_argument(
            '--min-params-to-wrap',
            type=int,
            metavar='D',
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=
            ('minimum number of params for a layer to be wrapped with FSDP() when '
             'training with --ddp-backend=fully_sharded. Smaller values will '
             'improve memory efficiency, but may make torch.distributed '
             'communication less efficient due to smaller input sizes. This option '
             'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
             '--offload-activations are passed.'))
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(','))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(','))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.vocab_src, task.vocab_tgt

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    '--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim'
                )
            if args.decoder_embed_path and \
                    (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embeddings not compatible with --decoder-embed-path'
                )
            encoder_embed_tokens = cls.build_embedding(args, src_dict,
                                                       args.encoder_embed_dim,
                                                       args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(args, src_dict,
                                                       args.encoder_embed_dim,
                                                       args.encoder_embed_path)
            decoder_embed_tokens = cls.build_embedding(args, tgt_dict,
                                                       args.decoder_embed_dim,
                                                       args.decoder_embed_path)
        if getattr(args, 'offload_activations', False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        second_decoder = cls.build_decoder(args, src_dict,
                                           encoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(args, 'min_params_to_wrap',
                                         DEFAULT_MIN_PARAMS_TO_WRAP)
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder, second_decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        prev_src_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        decoder_out_re = self.decoder(
            prev_output_tokens,
            encoder_out=None,
            features_only=features_only,
            full_context_alignment=True,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out_tensor = decoder_out_re[1]['last_layer']
        decoder_padding = decoder_out_re[1]['self_attn_padding_mask']

        decoder_kvs = {
            'encoder_out': [decoder_out_tensor],
            'encoder_padding_mask': [decoder_padding]
        }
        src_out = self.second_decoder(
            prev_src_tokens,
            encoder_out=decoder_kvs,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=None,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out, src_out, decoder_kvs

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs,
                                                    sample)

    def forward_decoder(
        self,
        tokens,
        encoder_outs: Dict[str, List[Tensor]],
        incremental_states: Dict[str, Dict[str, Optional[Tensor]]],
        temperature: float = 1.0,
    ):
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        encoder_out = encoder_outs
        # decode
        decoder_out = self.decoder.forward(
            tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_states,
        )

        attn: Optional[Tensor] = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]['attn']
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]

        decoder_out_tuple = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if decoder_len <= 1 else decoder_out[1],
        )
        probs = self.get_normalized_probs(
            decoder_out_tuple, log_probs=True, sample=None)
        probs = probs[:, -1, :]
        decoder_out_tensor = decoder_out[1]['last_layer']
        return probs, attn, decoder_out_tensor

    def forward_decoder_src(
        self,
        tokens,
        encoder_outs: Dict[str, List[Tensor]],
        incremental_states: Dict[str, Dict[str, Optional[Tensor]]],
        temperature: float = 1.0,
    ):
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        encoder_out = encoder_outs
        # decode each model
        decoder_out = self.second_decoder.forward(
            tokens, encoder_out=encoder_out)

        attn: Optional[Tensor] = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]['attn']
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]

        decoder_out_tuple = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if decoder_len <= 1 else decoder_out[1],
        )
        probs = self.get_normalized_probs(
            decoder_out_tuple, log_probs=True, sample=None)
        probs = probs[:, -1, :]
        decoder_out_tensor = decoder_out[1]['last_layer']
        return probs, attn, decoder_out_tensor, decoder_out

    def forward_encoder(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v
            for k, v in net_input.items() if k != 'prev_output_tokens'
            and k != 'prev_src_tokens' and k != 'sources'
        }
        return self.encoder.forward_torchscript(encoder_input)

    def reorder_encoder_out(self, encoder_outs: Optional[Dict[str,
                                                              List[Tensor]]],
                            new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        assert encoder_outs is not None
        return self.encoder.reorder_encoder_out(encoder_outs, new_order)

    def reorder_incremental_state(
        self,
        incremental_states: Dict[str, Dict[str, Optional[Tensor]]],
        new_order,
    ):
        self.decoder.reorder_incremental_state_scripting(
            incremental_states, new_order)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            ) if not args.no_token_positional_embeddings else None)
        export = getattr(args, 'export', False)
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            self.build_encoder_layer(args) for i in range(args.encoder_layers)
        ])
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        checkpoint = getattr(args, 'checkpoint_activations', False)
        if checkpoint:
            offload_to_cpu = getattr(args, 'offload_activations', False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = (
            getattr(args, 'min_params_to_wrap', DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0)
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(self,
                          src_tokens,
                          token_embedding: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens, src_lengths,
                                       return_all_hiddens, token_embeddings)

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == 'xla' or encoder_padding_mask.any(
        )
        x, encoder_embedding = self.forward_embedding(src_tokens,
                                                      token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask
                if has_pads else None)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            'encoder_out': [x],  # T x B x C
            'encoder_padding_mask': [encoder_padding_mask],  # B x T
            'encoder_embedding': [encoder_embedding],  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
            'src_tokens': [],
            'src_lengths': [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]],
                            new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out['encoder_out']) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [
                encoder_out['encoder_out'][0].index_select(1, new_order)
            ]
        if len(encoder_out['encoder_padding_mask']) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out['encoder_padding_mask'][0].index_select(
                    0, new_order)
            ]
        if len(encoder_out['encoder_embedding']) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out['encoder_embedding'][0].index_select(0, new_order)
            ]

        if len(encoder_out['src_tokens']) == 0:
            src_tokens = []
        else:
            src_tokens = [
                (encoder_out['src_tokens'][0]).index_select(0, new_order)
            ]

        if len(encoder_out['src_lengths']) == 0:
            src_lengths = []
        else:
            src_lengths = [
                (encoder_out['src_lengths'][0]).index_select(0, new_order)
            ]

        encoder_states = encoder_out['encoder_states']
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            'encoder_out': new_encoder_out,  # T x B x C
            'encoder_padding_mask': new_encoder_padding_mask,  # B x T
            'encoder_embedding': new_encoder_embedding,  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
            'src_tokens': src_tokens,  # B x T
            'src_lengths': src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions,
                   self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, '{}.layers.{}'.format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__)
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim else None)
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            ) if not args.no_token_positional_embeddings else None)
        export = getattr(args, 'export', False)
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, 'cross_self_attention',
                                            False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            self.build_decoder_layer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim
            and not args.tie_adaptive_weights else None)

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens
                if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False)
            nn.init.normal_(
                self.output_projection.weight,
                mean=0,
                std=self.output_embed_dim**-0.5)
        num_base_layers = getattr(args, 'base_layers', 0)
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder_layers) // (num_base_layers + 1),
                BaseLayer(args),
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, 'checkpoint_activations', False)
        if checkpoint:
            offload_to_cpu = getattr(args, 'offload_activations', False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = (
            getattr(args, 'min_params_to_wrap', DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0)
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str,
                                                   Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str,
                                                   Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str,
                                                   Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out['encoder_out']) > 0:
            enc = encoder_out['encoder_out'][0]
            assert (enc.size()[1] == bs
                    ), f'Expected enc.shape == (t, {bs}, c) got {enc.shape}'
        if encoder_out is not None and len(
                encoder_out['encoder_padding_mask']) > 0:
            padding_mask = encoder_out['encoder_padding_mask'][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(
                self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, self_attn_hidden = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        last_layer = x
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x, {
            'attn': [attn],
            'inner_states': inner_states,
            'last_layer': last_layer,
            'self_attn_padding_mask': self_attn_padding_mask
        }

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions,
                   self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)

        if f'{name}.output_projection.weight' not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f'{name}.embed_tokens.weight'
            else:
                embed_out_key = f'{name}.embed_out'
            if embed_out_key in state_dict:
                state_dict[f'{name}.output_projection.weight'] = state_dict[
                    embed_out_key]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm',
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(
                        name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(
                            name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim',
                                     args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim',
                                         args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout',
                                            0)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim',
                                      args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim',
                                     args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.checkpoint_activations = getattr(args, 'checkpoint_activations',
                                          False)
    args.offload_activations = getattr(args, 'offload_activations', False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.decoder_layers_to_keep = getattr(args, 'decoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0)
    args.decoder_layerdrop = getattr(args, 'decoder_layerdrop', 0)
    args.quant_noise_pq = getattr(args, 'quant_noise_pq', 0)
    args.quant_noise_pq_block_size = getattr(args, 'quant_noise_pq_block_size',
                                             8)
    args.quant_noise_scalar = getattr(args, 'quant_noise_scalar', 0)


def transformer_deep(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            True)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.01)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.01)
    args.dropout = getattr(args, 'dropout', 0.01)
    base_architecture(args)

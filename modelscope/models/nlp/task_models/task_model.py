# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import re
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch import nn

from modelscope.models.base import TorchModel
from modelscope.models.builder import build_backbone, build_head
from modelscope.outputs import OutputKeys
from modelscope.utils.checkpoint import load_task_model_checkpoint
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['EncoderModel', 'SingleBackboneTaskModelBase']


def _repr(modules, depth=1):
    # model name log level control
    if depth == 0:
        return modules._get_name()
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = modules.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []

    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    for key, module in modules._modules.items():
        mod_str = _repr(module, depth - 1)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = modules._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str


class BaseTaskModel(TorchModel, ABC):
    """ Base task model interface for nlp

    """
    # keys to ignore when load missing
    _keys_to_ignore_on_load_missing = None
    # keys to ignore when load unexpected
    _keys_to_ignore_on_load_unexpected = None
    # backbone prefix, default None
    _backbone_prefix = None

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.config = ConfigDict(kwargs)

    def __repr__(self):
        # only log backbone and head name
        depth = 1
        return _repr(self, depth)

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        model = cls(**kwargs)
        model.load_checkpoint(model_local_dir=model_dir, **kwargs)
        return model

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def load_checkpoint(self,
                        model_local_dir,
                        default_dtype=None,
                        load_state_fn=None,
                        **kwargs):
        """
        Load model checkpoint file and feed the parameters into the model.
        Args:
            model_local_dir: The actual checkpoint dir on local disk.
            default_dtype: Set the default float type by 'torch.set_default_dtype'
            load_state_fn: An optional load_state_fn used to load state_dict into the model.

        Returns:

        """
        # TODO Sharded ckpt
        ckpt_file = os.path.join(model_local_dir,
                                 ModelFile.TORCH_MODEL_BIN_FILE)
        state_dict = torch.load(ckpt_file, map_location='cpu')
        if default_dtype is not None:
            torch.set_default_dtype(default_dtype)

        missing_keys, unexpected_keys, mismatched_keys, error_msgs = self._load_checkpoint(
            state_dict,
            load_state_fn=load_state_fn,
            ignore_mismatched_sizes=True,
            _fast_init=True,
        )

        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'mismatched_keys': mismatched_keys,
            'error_msgs': error_msgs,
        }

    def _load_checkpoint(
        self,
        state_dict,
        load_state_fn,
        ignore_mismatched_sizes,
        _fast_init,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = self.state_dict()
        prefix = self._backbone_prefix

        # add head prefix
        new_state_dict = OrderedDict()
        for name, module in state_dict.items():
            if not name.startswith(prefix) and not name.startswith('head'):
                new_state_dict['.'.join(['head', name])] = module
            else:
                new_state_dict[name] = module
        state_dict = new_state_dict

        loaded_keys = [k for k in state_dict.keys()]
        expected_keys = list(model_state_dict.keys())

        def _fix_key(key):
            if 'beta' in key:
                return key.replace('beta', 'bias')
            if 'gamma' in key:
                return key.replace('gamma', 'weight')
            return key

        original_loaded_keys = loaded_keys
        loaded_keys = [_fix_key(key) for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(
                s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys_not_prefixed = [
                s for s in expected_keys if not s.startswith(prefix)
            ]
            expected_keys = [
                '.'.join(s.split('.')[1:]) if s.startswith(prefix) else s
                for s in expected_keys
            ]
        elif add_prefix_to_model:
            expected_keys = ['.'.join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        if self._keys_to_ignore_on_load_missing is not None:
            for pat in self._keys_to_ignore_on_load_missing:
                missing_keys = [
                    k for k in missing_keys if re.search(pat, k) is None
                ]

        if self._keys_to_ignore_on_load_unexpected is not None:
            for pat in self._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [
                    k for k in unexpected_keys if re.search(pat, k) is None
                ]

        if _fast_init:
            # retrieve uninitialized modules and initialize
            uninitialized_modules = self.retrieve_modules_from_names(
                missing_keys,
                prefix=prefix,
                add_prefix=add_prefix_to_model,
                remove_prefix=remove_prefix_from_model)
            for module in uninitialized_modules:
                self._init_weights(module)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = self
        if len(prefix) > 0 and not hasattr(self, prefix) and has_prefix_module:
            start_prefix = prefix + '.'
        if len(prefix) > 0 and hasattr(self, prefix) and not has_prefix_module:
            model_to_load = getattr(self, prefix)
            if any(key in expected_keys_not_prefixed for key in loaded_keys):
                raise ValueError(
                    'The state dictionary of the model you are trying to load is corrupted. Are you sure it was '
                    'properly saved?')

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f'{prefix}.{checkpoint_key}'
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = '.'.join(checkpoint_key.split('.')[1:])

                    if (model_key in model_state_dict):
                        model_shape = model_state_dict[model_key].shape
                        checkpoint_shape = state_dict[checkpoint_key].shape
                        if (checkpoint_shape != model_shape):
                            mismatched_keys.append(
                                (checkpoint_key,
                                 state_dict[checkpoint_key].shape,
                                 model_state_dict[model_key].shape))
                            del state_dict[checkpoint_key]
            return mismatched_keys

        def _load_state_dict_into_model(model_to_load, state_dict,
                                        start_prefix):
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            error_msgs = []

            if load_state_fn is not None:
                load_state_fn(
                    model_to_load,
                    state_dict,
                    prefix=start_prefix,
                    local_metadata=None,
                    error_msgs=error_msgs)
            else:

                def load(module: nn.Module, prefix=''):
                    local_metadata = {} if metadata is None else metadata.get(
                        prefix[:-1], {})
                    args = (state_dict, prefix, local_metadata, True, [], [],
                            error_msgs)
                    module._load_from_state_dict(*args)
                    for name, child in module._modules.items():
                        if child is not None:
                            load(child, prefix + name + '.')

                load(model_to_load, prefix=start_prefix)

            return error_msgs

        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )
        error_msgs = _load_state_dict_into_model(model_to_load, state_dict,
                                                 start_prefix)

        if len(error_msgs) > 0:
            error_msg = '\n\t'.join(error_msgs)
            raise RuntimeError(
                f'Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{error_msg}'
            )

        if len(unexpected_keys) > 0:
            logger.warning(
                f'Some weights of the model checkpoint were not used when'
                f' initializing {self.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are'
                f' initializing {self.__class__.__name__} from the checkpoint of a model trained on another task or'
                ' with another architecture (e.g. initializing a BertForTokenClassification model from a'
                ' BertForPreTraining model).\n- This IS NOT expected if you are initializing'
                f' {self.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical'
                ' (initializing a BertForTokenClassification model from a BertForTokenClassification model).'
            )
        else:
            logger.info(
                f'All model checkpoint weights were used when initializing {self.__class__.__name__}.\n'
            )
        if len(missing_keys) > 0:
            logger.warning(
                f'Some weights of {self.__class__.__name__} were not initialized from the model checkpoint'
                f' and are newly initialized: {missing_keys}\nYou should probably'
                ' TRAIN this model on a down-stream task to be able to use it for predictions and inference.'
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f'All the weights of {self.__class__.__name__} were initialized from the model checkpoint '
                f'If your task is similar to the task the model of the checkpoint'
                f' was trained on, you can already use {self.__class__.__name__} for predictions without further'
                ' training.')
        if len(mismatched_keys) > 0:
            mismatched_warning = '\n'.join([
                f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated'
                for key, shape1, shape2 in mismatched_keys
            ])
            logger.warning(
                f'Some weights of {self.__class__.__name__} were not initialized from the model checkpoint'
                f' and are newly initialized because the shapes did not'
                f' match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able'
                ' to use it for predictions and inference.')

        return missing_keys, unexpected_keys, mismatched_keys, error_msgs

    def retrieve_modules_from_names(self,
                                    names,
                                    prefix=None,
                                    add_prefix=False,
                                    remove_prefix=False):
        module_keys = set(['.'.join(key.split('.')[:-1]) for key in names])

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            set([
                '.'.join(key.split('.')[:-2]) for key in names
                if key[-1].isdigit()
            ]))

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                name = '.'.join(
                    name.split('.')[1:]) if name.startswith(prefix) else name
            elif add_prefix:
                name = '.'.join([prefix, name]) if len(name) > 0 else prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules


class SingleBackboneTaskModelBase(BaseTaskModel):
    """
    This is the base class of any single backbone nlp task classes.
    """
    # The backbone prefix defaults to "bert"
    _backbone_prefix = 'bert'

    # The head prefix defaults to "head"
    _head_prefix = 'head'

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.backbone_cfg = self.config.get('backbone', None)
        assert self.backbone_cfg is not None
        self.head_cfg = self.config.get('head', None)

    def build_backbone(self, cfg):
        if 'prefix' in cfg:
            self._backbone_prefix = cfg['prefix']
        backbone = build_backbone(cfg)
        setattr(self, cfg['prefix'], backbone)

    def build_head(self, cfg):
        if cfg is None:
            raise ValueError(
                'Head config is missing, check if this was a backbone-only model'
            )
        if 'prefix' in cfg:
            self._head_prefix = cfg['prefix']
        head = build_head(cfg, task_name=self.group_key)
        setattr(self, self._head_prefix, head)
        return head

    @property
    def backbone(self):
        if 'backbone' != self._backbone_prefix:
            return getattr(self, self._backbone_prefix)
        return super().__getattr__('backbone')

    @property
    def head(self):
        if 'head' != self._head_prefix:
            return getattr(self, self._head_prefix)
        return super().__getattr__('head')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """default forward method is the backbone-only forward"""
        if func_receive_dict_inputs(self.backbone.forward):
            outputs = self.backbone.forward(input)
        else:
            outputs = self.backbone.forward(**input)
        return outputs

    def compute_loss(self, outputs, labels):
        loss = self.head.compute_loss(outputs, labels)
        return loss

    def extract_backbone_outputs(self, outputs):
        sequence_output = None
        pooled_output = None
        if hasattr(self.backbone, 'extract_sequence_outputs'):
            sequence_output = self.backbone.extract_sequence_outputs(outputs)
        if hasattr(self.backbone, 'extract_pooled_outputs'):
            pooled_output = self.backbone.extract_pooled_outputs(outputs)
        return sequence_output, pooled_output


class EncoderModel(TorchModel):
    """
    This is the base class of any encoder nlp task classes.
    """
    # keys to ignore when load missing
    _keys_to_ignore_on_load_missing = None
    # keys to ignore when load unexpected
    _keys_to_ignore_on_load_unexpected = None
    # The encoder prefix defaults to "encoder"
    base_model_prefix = 'encoder'
    # The default backbone model type is None, should be bert/T5
    base_model_type = None
    # The head prefix defaults to "head"
    head_prefix = 'head'
    # The head type defaults as None
    head_type = None
    # override base model prefix by task model prefix
    override_base_model_prefix = False
    # override base model type by task model type
    override_base_model_type = False

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.config = ConfigDict(kwargs)
        backbone_cfg = self.parse_encoder_cfg()
        head_cfg = self.parse_head_cfg()
        self.build_encoder(backbone_cfg)
        if head_cfg.type is not None:
            self.build_head(head_cfg)

    def __repr__(self):
        # only log backbone and head name
        depth = 1
        return _repr(self, depth)

    def _get_transformer_config(self):
        transformer_config_file = os.path.join(self.model_dir,
                                               ModelFile.CONFIG)
        transformer_config = None
        if os.path.exists(transformer_config_file):
            transformer_config = Config.from_file(transformer_config_file)
        return transformer_config.copy()

    def _use_transformer_config(self, cfg):
        if 'model_type' not in cfg and 'type' not in cfg:
            return True
        else:
            return False

    def parse_encoder_cfg(self):
        # get encoder from backbone-head configuration format
        encoder_cfg = self.config.get('backbone', None)
        if encoder_cfg is None:
            encoder_cfg = self.config.copy()
            if 'model_type' in encoder_cfg and 'type' not in encoder_cfg:
                encoder_cfg.type = encoder_cfg.model_type
            elif self._use_transformer_config(encoder_cfg):
                encoder_cfg = self._get_transformer_config()
                encoder_cfg.type = encoder_cfg.model_type

        if 'type' not in encoder_cfg or self.override_base_model_type:
            encoder_cfg.type = self.base_model_type
        if encoder_cfg.type is None:
            raise KeyError(
                'Missing encoder type, please explicit define encoder type in configuration.json'
            )
        encoder_cfg.model_dir = self.model_dir
        return encoder_cfg

    def parse_head_cfg(self):
        head_cfg = self.config.get('head', None)
        if head_cfg is None:
            head_cfg = self.config.copy()
            if 'head_type' in head_cfg and 'type' not in head_cfg:
                head_cfg.type = head_cfg.head_type
            elif self._use_transformer_config(head_cfg):
                head_cfg = self._get_transformer_config()
                head_cfg.type = self.head_type
        if 'type' not in head_cfg:
            head_cfg.type = self.head_type
        return head_cfg

    def build_encoder(self, cfg):
        backbone = build_backbone(cfg)
        if 'prefix' in cfg:
            self.base_model_prefix = cfg['prefix']
        elif 'base_model_prefix' in cfg:
            self.base_model_prefix = cfg['base_model_prefix']
        elif hasattr(backbone, 'base_model_prefix') \
                and not self.override_base_model_prefix:
            self.base_model_prefix = backbone.base_model_prefix
        setattr(self, self.base_model_prefix, backbone)

    def build_head(self, cfg):
        if cfg is None:
            raise ValueError(
                'Head config is missing, check if this was a backbone-only model'
            )
        head = build_head(cfg, task_name=self.group_key)
        setattr(self, self.head_prefix, head)

    @property
    def encoder(self):
        if 'encoder' != self.base_model_prefix:
            return getattr(self, self.base_model_prefix)
        return super().__getattr__('encoder')

    @property
    def head(self):
        if 'head' != self.head_prefix:
            return getattr(self, self.head_prefix)
        return super().__getattr__('head')

    def extract_feature(self, **input: Dict[str, Any]) -> Dict[str, Any]:
        """default forward method is the backbone-only forward"""
        if func_receive_dict_inputs(self.encoder.forward):
            outputs = self.encoder.forward(input)
        else:
            outputs = self.encoder.forward(**input)
        return outputs

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                *args,
                **kwargs):
        r"""
        Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~modelscope.models.nlp.structbert.SbertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
         output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.ModelOutput` instead of a plain tuple.
        *args:
            In Torch 1.11 onnx has a bug in the _slow_forward method, could only keep *args solving the problem
        **kwargs:
            Accept additional kwargs in the children class

        Returns:
            Returns `modelscope.outputs.ModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
            >>> print(model(**preprocessor(('This is a test', 'This is also a test'))))
        """

        if OutputKeys.LABEL in kwargs and labels is None:
            labels = kwargs.pop(OutputKeys.LABEL, None)
        feature = self.extract_feature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs = self.head.forward(feature, attention_mask, labels, **kwargs)
        return outputs

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        model = cls(**kwargs)
        model_load_handler = load_task_model_checkpoint(
            model_to_load=model, model_local_dir=model_dir, **kwargs)
        return model_load_handler['model']

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        cfg_dict: Config = None,
                        device: str = None,
                        **kwargs):
        task = kwargs.pop('task', None)
        return super(TorchModel, cls).from_pretrained(
            model_name_or_path=model_name_or_path,
            revision=revision,
            cfg_dict=cfg_dict,
            devic=device,
            task=task if task is not None else cls.task,
            **kwargs)

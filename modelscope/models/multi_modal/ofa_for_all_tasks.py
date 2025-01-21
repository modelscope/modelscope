# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import re
import string
from functools import partial
from os import path as osp
from typing import Any, Callable, Dict, List, Optional, Union

import json
import torch.cuda
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.ofa.utils.collate import collate_tokens
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.trie import Trie
from .ofa import MMSpeechModel, OFAModel, OFATokenizer, OFATokenizerZH
from .ofa.generate import sequence_generator as sg
from .ofa.generate.utils import move_to_device
from .ofa.utils.constant import OFA_TASK_KEY_MAPPING, Tasks
from .ofa.utils.utils import expand_mask

__all__ = ['OfaForAllTasks']


@MODELS.register_module(Tasks.image_captioning, module_name=Models.ofa)
@MODELS.register_module(Tasks.ocr_recognition, module_name=Models.ofa)
@MODELS.register_module(Tasks.visual_grounding, module_name=Models.ofa)
@MODELS.register_module(
    Tasks.visual_question_answering, module_name=Models.ofa)
@MODELS.register_module(Tasks.visual_entailment, module_name=Models.ofa)
@MODELS.register_module(Tasks.image_classification, module_name=Models.ofa)
@MODELS.register_module(Tasks.text_summarization, module_name=Models.ofa)
@MODELS.register_module(Tasks.text_classification, module_name=Models.ofa)
@MODELS.register_module(Tasks.auto_speech_recognition, module_name=Models.ofa)
@MODELS.register_module(Tasks.sudoku, module_name=Models.ofa)
@MODELS.register_module(Tasks.text2sql, module_name=Models.ofa)
class OfaForAllTasks(TorchModel):
    r"""
    All ofa tasks using uniform ofa model structure. So far, we support three types of tasks:
        1. text generation tasks: ocr_recognition, image_captioning and text_summarization
        2. visual grounding tasks: visual grounding
        3. classification tasks: text classification and image classification.

    Attributes:
        cfg: Task configs exclude model configs, such as generator's config.
        model:  OFA uniform model using in this task.
        language: The language using in the model. So far, we support three types of language, `en` for English,
                `zh` and `cn` for Chinese, default to `en`.
        tokenizer: OFA tokenizer for tokenizing the input for OFA model.
        batch_size: Batch size.
        patch_image_size: The image size of input image, default to 480.
        val_batch_size: The validation batch size.
        transtab: A translation table of punctuation.
        gen_type: Generation type, so far, we support two types of gen_type, `generation` for generation tasks,
                 `traverse` for classification tasks, default to `generation`.
        bos_item: The id of beginning of a sequence.
        pad_item: The id of padding of a sequence.
        eos_item: The id of ending of a sequence.
        index2ans: A mapping from index to label using in classification tasks.
        ans2label_dict: A mapping from label to index using in classification tasks.
        constraint_trie: A trie tree building from label using in classification tasks.
        val_ans_l: A validation set of label using in classification tasks.
        val_masks_l: A validation set of mask using in classification tasks.
        generator: A sequence generator with OFA model to generate image code.
        task_inference_mapping: A mapping from task name to execution function in task inference.
        pattern: Regex pattern which find the blanks after/before the words except ` a-zA-Z0-9.,:!?`
    """

    def __init__(self, model_dir, *args, **kwargs):
        r"""
        Args:
            model_dir (`str` or `os.PathLike`)
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co
                      or modelscope.cn. Valid model ids can be located at the root-level, like `bert-base-uncased`,
                      or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
        """
        if os.path.exists(model_dir):
            model_dir = os.path.abspath(model_dir)
        super().__init__(model_dir=model_dir, *args, **kwargs)
        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        multimodal_type = self.cfg.model.get('multimodal_type', 'default')
        if multimodal_type in ['default', 'text2sql']:
            model = OFAModel.from_pretrained(model_dir)
        elif multimodal_type == 'mmspeech':
            model = MMSpeechModel.from_pretrained(model_dir)
        else:
            raise NotImplementedError
        self.model = model.module if hasattr(model, 'module') else model
        self.language = self.cfg.model.get('language', 'en')
        if self.language == 'en':
            self.tokenizer = OFATokenizer.from_pretrained(model_dir)
        elif self.language in ['zh', 'cn']:
            self.tokenizer = OFATokenizerZH.from_pretrained(model_dir)
        else:
            raise NotImplementedError
        # there is some diff between here and our ofa code,
        # there will be no need to use param: use_bpe

        if not model.use_ofasys:
            if multimodal_type == 'default':
                self.tokenizer.add_tokens(
                    ['<code_{}>'.format(i) for i in range(8192)])
                self.tokenizer.add_tokens(
                    ['<bin_{}>'.format(i) for i in range(1000)])
                self.cfg.update({'num_bins': 1000, 'num_codes': 8192})
            elif multimodal_type == 'mmspeech':
                self.tokenizer.add_tokens('<blank>')
                self.tokenizer.add_tokens(
                    ['<audio_{}>'.format(i) for i in range(30000)])
                self.cfg.update({'num_bins': 0, 'num_codes': 30000})
            elif multimodal_type == 'text2sql':
                self.tokenizer.add_tokens(
                    ['<code_{}>'.format(i) for i in range(8192)])
                self.tokenizer.add_tokens(
                    ['<bin_{}>'.format(i) for i in range(1000)])
                self.cfg.update({'num_bins': 1000, 'num_codes': 8192})
                self.tokenizer.add_tokens(['>=', '<='])

        self.batch_size = self.cfg.model.get('batch_size', 1)
        self.patch_image_size = self.cfg.model.get('patch_image_size', 480)
        self.max_image_size = self.cfg.model.get('max_image_size', 512)
        self.val_batch_size = self.cfg.model.get('valid_batch_size',
                                                 self.batch_size)
        self.transtab = str.maketrans(
            {key: None
             for key in string.punctuation})
        self.gen_type = self.cfg.model.get('gen_type', 'generation')
        assert self.gen_type in ['generation', 'traverse'], \
            'model.gen_type must be in ["generation", "traverse"]'
        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.pad_item = torch.LongTensor([self.tokenizer.pad_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.index2ans = {}
        self.ans2label_dict = {}
        self.load_ans2label()
        # Initialize generator
        sg_args = {
            'tokenizer': self.tokenizer,
            'beam_size': 5,
            'max_len_b': 16,
            'min_len': 1,
            'no_repeat_ngram_size': 3,
            'constraint_range': None
        }
        if hasattr(self.cfg.model, 'beam_search'):
            sg_args.update(self.cfg.model.beam_search)
        self.num_return_sequences = self.cfg.model.get('num_return_sequences',
                                                       1)
        if len(self.ans2label_dict) > 0:
            self.constraint_trie = Trie(self.tokenizer.eos_token_id)
            self.val_ans_l = []
            self.val_masks_l = []
            self.build_trie()
            sg_args['constraint_trie'] = self.constraint_trie
        else:
            self.constraint_trie = None
        self.generator = sg.SequenceGenerator(**sg_args)
        inference_d = {
            'generation': self._text_gen_inference,
            'traverse': self._traverse_inference,
        }
        self.task_inference_mapping = {
            Tasks.ocr_recognition: self._text_gen_inference,
            Tasks.image_captioning: self._text_gen_inference,
            Tasks.text_summarization: self._text_gen_inference,
            Tasks.visual_grounding: self._visual_grounding_inference,
            Tasks.visual_entailment: inference_d[self.gen_type],
            Tasks.visual_question_answering: inference_d[self.gen_type],
            Tasks.text_classification: inference_d[self.gen_type],
            Tasks.image_classification: inference_d[self.gen_type],
            Tasks.auto_speech_recognition: self._text_gen_inference,
            Tasks.sudoku: self._text_gen_inference,
            Tasks.text2sql: self._text_gen_inference,
        }
        pattern_str = '((?<=[^ a-zA-Z0-9.,:!?]) +| +(?=[^ a-zA-Z0-9.,:!?]))'
        self.pattern = re.compile(pattern_str)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        The entry function of task execution. So far, we support two types of execution pipeline:
        1. training, return the model's forward results.
        2. inference, return the result of `self.inference(input)`

        Args:
            input (`Dict[Str, Any]`):
                The input of the tasks, the actual value depending on the specific tasks.
        Returns:
            `Dict[Str, Any]`

        """
        input = move_to_device(input, self.model.device)
        if self.model.training:
            return self.model(**input['net_input'])
        else:
            return self.inference(input)

    def inference(self, input: Dict[str, Any]) -> Dict[str, Any]:
        assert self.generator.beam_size >= self.num_return_sequences, \
            'beam search can only return beam size sentences'
        if self.ans2label_dict and self.gen_type == 'generation':
            assert self.generator.beam_size <= len(self.ans2label_dict), \
                'beam search will not work properly.'
        r"""
        Task inference function

        Args:
            input (`Dict[Str, Any]`):
                The input of the tasks, the actual value depending on the specific tasks.
        Returns:
            `Dict[Str, Any]`

        """
        ret = self.task_inference_mapping[self.cfg.task](input)
        if 'samples' in input:
            ret['samples'] = input['samples']
        return ret

    def postprocess(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        r"""
        Do post processing after task's forward function is executed. So far, we have three strategies while do post
            processing.

            1. If the task is image captioning and using English language, some special words will be removed, such as
               `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`
            2. If the task is not visual grounding, but a generation task using Chinese language, we will remove the
                blank after/before the words except ` a-zA-Z0-9.,:!?`
            3. Other cases will return the input as result.

        Args:
            input (`Dict[Str, Any]`):
                The result of task's forward function. The key is one of the keys of OFA_TASK_KEY_MAPPING for
                distinguishing different ofa tasks, while the value is the result of different tasks.

        Returns:
            `Dict[Str, Any]`
        """
        if not self.model.training and self.cfg.task == Tasks.image_captioning:
            caption = input[OutputKeys.CAPTION]
            result_l = list()
            for cap in caption:
                if self.language == 'en':
                    result_l.append(
                        [c.translate(self.transtab).strip() for c in cap])
                else:
                    result_l.append(cap)
            input[OutputKeys.CAPTION] = result_l
        if self.gen_type == 'generation' and self.language in [
                'zh', 'cn'
        ] and self.cfg.task != Tasks.visual_grounding:
            ret_l = list()
            for text in input[OFA_TASK_KEY_MAPPING[self.cfg.task]]:
                ret_l.append([self.detokenizer(t) for t in text])
            input[OFA_TASK_KEY_MAPPING[self.cfg.task]] = ret_l
        for key in [
                OutputKeys.CAPTION, OutputKeys.TEXT, OutputKeys.BOXES,
                OutputKeys.LABELS, OutputKeys.SCORES
        ]:
            if key not in input:
                input[key] = None
            else:
                if (len(input[key]) == 1 and isinstance(input[key], list)) \
                        and self.cfg.task != Tasks.visual_grounding:
                    input[key] = input[key][0]
        return input

    def _text_gen_inference(self, input):
        r"""
        The inference function fo text generation tasks.
        1. Using OFA sequence generator which match the api of other fairseq generators to generate the token indices.
        2. Decode the token indices to actual language tokens and skip the special tokens.
        3. For the usage of classification scenario, add default score with `len(result)`.

        Args:
            input (`Dict[Str, Any]`):
                The input of the tasks, the actual value depending on the specific tasks.
        Returns:
            `Dict[Str, Any]`
        """
        gen_outputs = self.generator.generate([self.model],
                                              input,
                                              prefix_tokens=input.get(
                                                  'prefix_tokens', None))
        results = list()
        for idx, gen_out in enumerate(gen_outputs):
            gen_token_l = []
            for beam_gen_out in gen_out[:self.num_return_sequences]:
                decode_tokens = beam_gen_out['tokens']
                if 'prefix_tokens' in input:
                    prefix_len = input['prefix_tokens'][idx].ne(
                        self.pad_item.to(self.model.device)).sum()
                    decode_tokens = decode_tokens[prefix_len:]
                gen_token_l.append(decode_tokens)
            result = self.tokenizer.batch_decode(
                gen_token_l, skip_special_tokens=True)
            result = [item.strip() for item in result]
            result.extend([''] * (self.num_return_sequences - len(result)))
            results.append(result)
        # text generation tasks have no score
        ret = {OFA_TASK_KEY_MAPPING[self.cfg.task]: results}
        if self.ans2label_dict:
            ret[OutputKeys.SCORES] = [[1.0]] * len(results)
        return ret

    def _visual_grounding_inference(self, input):
        r"""
        The inference function for visual grounding tasks.
        1. Using OFA sequence generator which match the api of other fairseq generators to generate the token indices.
        2. Decode the token indices into region boxes.
        3. Add default score with `batch_size`

        Args:
            input (`Dict[Str, Any]`):
                The input of the tasks, the actual value depending on the specific tasks.
        Returns:
            `Dict[Str, Any]`
        """
        gen_output = self.generator.generate([self.model], input)
        tokens = [gen_output[i][0]['tokens'] for i in range(len(gen_output))]
        region_coord_l = list()
        for i in range(len(tokens)):
            region_coord_l.append(tokens[i][:-1]
                                  - len(self.tokenizer.get_vocab().items())
                                  + self.cfg.num_bins)
        region_tensor = torch.stack(region_coord_l, dim=0)
        region_tensor = region_tensor / (self.cfg.num_bins
                                         - 1) * self.max_image_size
        region_tensor[:, ::2] /= input['w_resize_ratios']
        region_tensor[:, 1::2] /= input['h_resize_ratios']
        return {
            OutputKeys.BOXES:
            move_to_device(region_tensor, torch.device('cpu')).tolist(),
            OutputKeys.SCORES: [1.0] * region_tensor.shape[0]
        }

    def _traverse_inference(self, input):
        r"""
        The inference function fo classification tasks.

        Args:
            input (`Dict[Str, Any]`):
                The input of the tasks, the actual value depending on the specific tasks.
        Returns:
            `Dict[Str, Any]`
        """
        encoder_input = dict()
        for key in input['net_input'].keys():
            encoder_input[key] = input['net_input'][key]
        encoder_out = self.model.encoder(**encoder_input)
        valid_result = []
        for val_ans, val_masks in zip(self.val_ans_l, self.val_masks_l):
            valid_size = len(val_ans)
            valid_tgt_items = [
                torch.cat([
                    torch.tensor(decoder_prompt[1:]).to('cpu'), valid_answer,
                    self.eos_item
                ]) for decoder_prompt in input['decoder_prompts']
                for valid_answer in val_ans
            ]
            valid_prev_items = [
                torch.cat(
                    [torch.tensor(decoder_prompt).to('cpu'), valid_answer])
                for decoder_prompt in input['decoder_prompts']
                for valid_answer in val_ans
            ]
            valid_constraint_mask_items = [
                torch.cat([
                    torch.zeros(
                        len(decoder_prompt) - 1,
                        valid_constraint_mask.size(1)).bool(),
                    valid_constraint_mask], dim=0)  # yapf: disable
                for decoder_prompt in input['decoder_prompts']  # yapf: disable
                for valid_constraint_mask in val_masks]  # yapf: disable
            valid_tgt = collate_tokens(
                valid_tgt_items,
                pad_idx=self.tokenizer.pad_token_id).to(self.model.device)
            valid_prev_output = collate_tokens(
                valid_prev_items,
                pad_idx=self.tokenizer.pad_token_id).to(self.model.device)
            val_masks = collate_tokens(
                valid_constraint_mask_items,
                pad_idx=self.tokenizer.pad_token_id).to(self.model.device)
            new_encoder_out = {
                'last_hidden_state':
                encoder_out['last_hidden_state'].repeat_interleave(
                    valid_size, dim=0),
                'padding_mask':
                encoder_out['padding_mask'].repeat_interleave(
                    valid_size, dim=0),
                'position_embedding':
                encoder_out['position_embedding'].repeat_interleave(
                    valid_size, dim=0)
            }
            encoder_attention_mask = expand_mask(
                new_encoder_out['padding_mask'],
                new_encoder_out['last_hidden_state'].dtype,
                valid_prev_output.shape[-1])

            decoder_out = self.model.decoder(
                valid_prev_output,
                encoder_hidden_states=new_encoder_out['last_hidden_state'],
                encoder_attention_mask=encoder_attention_mask,
                src_pos_embed=new_encoder_out['position_embedding'])

            decoder_out[0].masked_fill_(~val_masks, -math.inf)
            lprobs = self.model.get_normalized_probs(
                decoder_out, log_probs=True)
            scores = lprobs.gather(
                dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(
                valid_tgt.eq(self.tokenizer.pad_token_id), 0)
            scores = scores.masked_fill((~val_masks).all(2), 0)
            scores = scores.sum(1)
            scores = scores.view(-1, valid_size)
            valid_result.append(scores)
        valid_result = torch.cat(valid_result, dim=-1)
        predicts = valid_result.argmax(1).tolist()
        probs = F.softmax(valid_result, dim=-1)
        hyps = [self.index2ans[predict_index] for predict_index in predicts]
        scores = [
            float(prob[idx].cpu().detach().numpy())
            for prob, idx in zip(probs, predicts)
        ]
        return {OutputKeys.LABELS: hyps, OutputKeys.SCORES: scores}

    def build_trie(self):
        r"""
        Building a trie tree for classification label and mask.
        """
        answer_item_list = []

        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tokenizer(
                ' ' + answer, return_tensors='pt',
                add_special_tokens=False).input_ids.squeeze(0)
            answer_item_list.append(answer_item)
            self.index2ans[i] = answer
            self.constraint_trie.insert([self.tokenizer.bos_token_id]
                                        + answer_item.tolist()
                                        + [self.tokenizer.eos_token_id])

        constraint_mask_list = []
        for answer_item in answer_item_list:
            constraint_mask = torch.zeros(
                (len(answer_item) + 1,
                 len(self.tokenizer.get_vocab()))).bool()
            for i in range(len(answer_item) + 1):
                constraint_prefix_token = [self.tokenizer.bos_token_id
                                           ] + answer_item[:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(
                    constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        for i in range(0, len(answer_item_list), self.val_batch_size):
            self.val_ans_l += [answer_item_list[i:i + self.val_batch_size]]
            self.val_masks_l += [
                constraint_mask_list[i:i + self.val_batch_size]
            ]

    def load_ans2label(self):
        r"""
        Load answer to label dict from file, using in building trie function.
        """
        if self.cfg.model.get('answer2label', None):
            ans2label_file = osp.join(self.model_dir,
                                      self.cfg.model.answer2label)
            with open(ans2label_file, 'r', encoding='utf-8') as reader:
                self.ans2label_dict = json.load(reader)

    def save_pretrained(self,
                        target_folder: Union[str, os.PathLike],
                        save_checkpoint_names: Union[str, List[str]] = None,
                        save_function: Callable = None,
                        config: Optional[dict] = None,
                        **kwargs):
        r"""
        Save the task model, its configuration and other related files to a directory, so that it can be re-loaded

        Args:
            target_folder (Union[str, os.PathLike]):
            Directory to which to save. Will be created if it doesn't exist.

            save_checkpoint_names (Union[str, List[str]]):
            The checkpoint names to be saved in the target_folder

            save_function (Callable, optional):
            The function to use to save the state dictionary.

            config (Optional[dict], optional):
            The config for the configuration.json, might not be identical with model.config

        """
        super(OfaForAllTasks, self). \
            save_pretrained(target_folder=target_folder,
                            save_checkpoint_names=save_checkpoint_names,
                            save_function=partial(save_function, with_meta=False),
                            config=config,
                            **kwargs)

    def detokenizer(self, text):
        r"""
        Remove the blank after/before the words except ` a-zA-Z0-9.,:!?`
        """
        return self.pattern.sub('', text)

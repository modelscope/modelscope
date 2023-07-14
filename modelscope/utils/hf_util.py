# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from transformers import AutoConfig as AutoConfigHF
from transformers import AutoModel as AutoModelHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import GenerationConfig as GenerationConfigHF

from modelscope import snapshot_download


class AutoModel(AutoModelHF):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path, revision=revision)
        else:
            model_dir = pretrained_model_name_or_path

        return super().from_pretrained(model_dir, *model_args, **kwargs)


class AutoModelForCausalLM(AutoModelForCausalLMHF):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path, revision=revision)
        else:
            model_dir = pretrained_model_name_or_path

        return super().from_pretrained(model_dir, *model_args, **kwargs)


class AutoModelForSeq2SeqLM(AutoModelForSeq2SeqLMHF):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path, revision=revision)
        else:
            model_dir = pretrained_model_name_or_path

        return super().from_pretrained(model_dir, *model_args, **kwargs)


class AutoTokenizer(AutoTokenizerHF):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.bin']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return super().from_pretrained(model_dir, *model_args, **kwargs)


class AutoConfig(AutoConfigHF):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.bin', r'\w+\.py']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return super().from_pretrained(model_dir, *model_args, **kwargs)


class GenerationConfig(GenerationConfigHF):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        ignore_file_pattern = [r'\w+\.bin', r'\w+\.py']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return super().from_pretrained(model_dir, *model_args, **kwargs)

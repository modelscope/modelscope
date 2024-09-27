import re
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from modelscope import AutoTokenizer, get_logger, snapshot_download
from . import TemplateType
from .base import Template, get_template

logger = get_logger()


@dataclass
class TemplateInfo:

    template: str = None
    template_regex: str = None
    modelfile_link: str = None


def cases(*names):
    ret = []
    for name in names:
        regex = ''
        for letter in name:
            if letter.upper() != letter.lower():
                regex += f'[{letter.upper()}{letter.lower()}]'
            else:
                regex += letter
        ret.append(regex)
    if len(ret) > 1:
        ret = '|'.join(ret)
        ret = '(' + ret + ')'
    else:
        ret = ret[0]
    return ret


chat_suffix = cases('instruct', 'chat', '-rl', '-it')


def no(*names):
    return f'(?!.*{cases(*names)})'


def no_multi_modal():
    return no('audio', 'video', 'vl', 'vision')


template_info = [
    # llama
    TemplateInfo(
        template=TemplateType.llama3,
        template_regex=
        f'.*{cases("llama3", "llama-3-")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama-3.modelfile',
    ),
    TemplateInfo(
        template=TemplateType.llama,
        template_regex=
        f'{cases("llama", "llama-2-", "mistral", "codestral", "mixtral")}{no_multi_modal()}.*{chat_suffix}.*'
    ),

    # qwen
    TemplateInfo(
        template=TemplateType.qwen,
        template_regex=f'.*{cases("qwen")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/qwen2.modelfile',
    ),

    # codeqwen1.5
    TemplateInfo(
        template_regex=
        f'.*{cases("codeqwen1.5", "codeqwen-1.5-")}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/codeqwen1.5.modelfile',
    ),

    # chatml
    TemplateInfo(
        template=TemplateType.chatml,
        template_regex=
        f'.*{cases("yi")}{no_multi_modal()}{no("coder")}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/yi-1.5.modelfile',
    ),

    # chatml
    TemplateInfo(
        template=TemplateType.chatml,
        template_regex=f'.*{cases("minicpm")}{no("-v")}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/yi-1.5.modelfile'
    ),

    # chatglm
    TemplateInfo(
        template=TemplateType.chatglm2,
        template_regex=f'.*{cases("chatglm2")}{no_multi_modal()}.*'),
    TemplateInfo(
        template=TemplateType.chatglm3,
        template_regex=f'.*{cases("chatglm3")}{no_multi_modal()}.*'),
    TemplateInfo(
        template=TemplateType.chatglm4,
        template_regex=f'.*{cases("glm4")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/glm4.modelfile',
    ),

    # baichuan
    TemplateInfo(
        template=TemplateType.baichuan,
        template_regex=
        f'.*{cases("baichuan")}{no_multi_modal()}.*{chat_suffix}.*'),

    # codegeex
    TemplateInfo(
        template=TemplateType.codegeex4,
        template_regex=f'.*{cases("codegeex4")}{no_multi_modal()}.*'),

    # idefics3
    TemplateInfo(
        template=TemplateType.idefics3,
        template_regex=f'.*{cases("idefics3")}{no_multi_modal()}.*'),

    # mistral-nemo
    TemplateInfo(
        template=TemplateType.mistral_nemo,
        template_regex=f'.*{cases("Mistral-Nemo")}{no_multi_modal()}.*',
        modelfile_link='https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistral-nemo.modelfile'),

    # internlm
    TemplateInfo(
        template=TemplateType.internlm,
        template_regex=
        f'.*{cases("internlm")}{no("internlm2", "internlm3")}{no_multi_modal()}.*{chat_suffix}.*'
    ),

    # internlm2
    TemplateInfo(
        template=TemplateType.internlm2,
        template_regex=
        f'.*{cases("internlm2")}{no_multi_modal()}.*{chat_suffix}.*'),

    # yi-coder
    TemplateInfo(
        template=TemplateType.yi_coder,
        template_regex=f'.*{cases("yi")}.*{cases("coder")}.*{chat_suffix}.*'),

    # yuan
    TemplateInfo(
        template=TemplateType.yuan,
        template_regex=f'.*{cases("Yuan")}{no_multi_modal()}.*'),

    # xverse
    TemplateInfo(
        template=TemplateType.xverse,
        template_regex=f'.*{cases("xverse")}{no_multi_modal()}.*{chat_suffix}.*'
    ),

    # skywork
    TemplateInfo(
        template=TemplateType.skywork,
        template_regex=
        f'.*{cases("skywork")}{no_multi_modal()}.*{chat_suffix}.*'),

    # bluelm
    TemplateInfo(
        template=TemplateType.bluelm,
        template_regex=f'.*{cases("bluelm")}{no_multi_modal()}.*{chat_suffix}.*'
    ),

    # zephyr
    TemplateInfo(
        template=TemplateType.zephyr,
        template_regex=f'.*{cases("zephyr")}{no_multi_modal()}.*'),

    # deepseek
    TemplateInfo(
        template=TemplateType.deepseek,
        template_regex=
        f'.*{cases("deepseek")}{no("v2", "v2.5", "coder")}{no_multi_modal()}.*{chat_suffix}.*'
    ),

    # deepseek2
    TemplateInfo(
        template=TemplateType.deepseek2,
        template_regex=
        f'.*{cases("deepseek")}.*{cases("v2")}{no("v2.5")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek_v2.modelfile',
    ),

    # deepseek_coder
    TemplateInfo(
        template=TemplateType.deepseek_coder,
        template_regex=
        f'.*{cases("deepseek")}{no("v2", "v2.5")}.*{cases("coder")}.*{chat_suffix}.*'
    ),

    # deepseek v2.5
    TemplateInfo(
        template=TemplateType.deepseek2_5,
        template_regex=
        f'.*{cases("deepseek")}.*{cases("v2.5")}{no_multi_modal()}.*'),

    # orion
    TemplateInfo(
        template=TemplateType.orion,
        template_regex=f'.*{cases("orion")}{no_multi_modal()}.*{chat_suffix}.*'
    ),

    # gemma
    TemplateInfo(
        template=TemplateType.gemma,
        template_regex=
        f'{no("pali")}.*{cases("gemma2", "gemma-2-")}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/gemma2.modelfile',
    ),

    # phi3
    TemplateInfo(
        template=TemplateType.phi3,
        template_regex=
        f'.*{cases("phi3", "phi-3-")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_link=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/phi3.modelfile',
    ),

    # telechat
    TemplateInfo(
        template=TemplateType.telechat,
        template_regex=f'.*{cases("TeleChat")}{no("v2")}.*'),

    # telechat_v2
    TemplateInfo(
        template=TemplateType.telechat_v2,
        template_regex=f'.*{cases("TeleChat")}.*{cases("v2")}.*'),
]


class TemplateLoader:

    @staticmethod
    def load_by_model_id(model_id: str, **kwargs) -> Template:
        """Load a template by model-id

        Args:
            model_id: The model-id used to load the proper template
            kwargs:
                revision: the revision of the model, default is `master`
        Returns:
            The template instance
        """
        ignore_file_pattern = [r'.+\.bin$', r'.+\.safetensors$', r'.+\.gguf$']
        tokenizer = kwargs.get('tokenizer')
        for _info in template_info:
            if re.fullmatch(_info.template_regex, model_id):
                if _info.template:
                    if tokenizer is None:
                        try:
                            model_dir = snapshot_download(
                                model_id,
                                revision=kwargs.pop('revision', 'master'),
                                ignore_file_pattern=ignore_file_pattern)
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_dir, trust_remote_code=True)
                        except Exception:
                            pass
                    return TemplateLoader.load_by_template_name(
                        _info.template, tokenizer=tokenizer, **kwargs)

    @staticmethod
    def load_by_template_name(template_name: str, **kwargs) -> Template:
        """Load a template by model-id

        Args:
            template_name: The template name used to load the proper template
            kwargs:
                tokenizer: The tokenizer of the model
                default_system: The extra default system info
                max_length: The max_length for the sequence
                truncation_strategy: 'delete' or 'truncation_left' the sequence of the length exceeds the limit
        Returns:
            The template instance
        """
        return get_template(template_name, tokenizer=kwargs.pop('tokenizer', None), **kwargs)

    @staticmethod
    def replace_and_concat(template: Template, template_list: List,
                           placeholder: str, keyword: str):
        final_str = ''
        for t in template_list:
            if isinstance(t, str):
                final_str += t.replace(placeholder, keyword)
            elif isinstance(t, (tuple, list)):
                if isinstance(t[0], int):
                    final_str += template.tokenizer.decode(t)
                else:
                    for attr in t:
                        if attr == 'bos_token_id':
                            final_str += template.tokenizer.bos_token
                        elif attr == 'eos_token_id':
                            final_str += template.tokenizer.eos_token
                        else:
                            raise ValueError(f'Unknown token: {attr}')
        return final_str

    @staticmethod
    def to_ollama(model_id: str = None,
                  template_name: str = None,
                  gguf_file: str = None,
                  gguf_meta: Dict[str, Any] = None,
                  **kwargs) -> str:
        """Export to ollama ModelFile

        Args:
            model_id: The model-id to use
            template_name: An extra template name to use
            gguf_file: An extra gguf_file path to use in the `FROM` field
            gguf_meta: An gguf extra meta info
        Returns:
            The ModelFile content, returns `None` if no template found
        """
        logger.info('Exporting to ollama:')
        if model_id:
            for _info in template_info:
                if re.fullmatch(_info.template_regex, model_id):
                    if _info.modelfile_link:
                        return TemplateLoader._read_content_from_url(
                            _info.modelfile_link)
                    elif _info.template and not template_name:
                        template_name = _info.template
        if template_name:
            template = TemplateLoader.load_by_template_name(
                template_name, **kwargs)
        else:
            raise ValueError(
                f'Please make sure you model_id: {model_id} '
                f'and template_name: {template_name} is supported.')

        if template is None:
            return None

        content = ''
        content += 'FROM {{gguf_file}}\n'
        content += (
            f'TEMPLATE """{{{{ if .System }}}}'
            f'{TemplateLoader.replace_and_concat(template, template.system_prefix or [], "{{SYSTEM}}", "{{ .System }}")}'
            f'{{{{ else }}}}{TemplateLoader.replace_and_concat(template, template.prefix, "", "")}'
            f'{{{{ end }}}}')
        content += (
            f'{{{{ if .Prompt }}}}'
            f'{TemplateLoader.replace_and_concat(template, template.prompt, "{{QUERY}}", "{{ .Prompt }}")}'
            f'{{{{ end }}}}')
        content += '{{ .Response }}'
        content += TemplateLoader.replace_and_concat(template, template.suffix,
                                                     '', '') + '"""\n'
        content += f'PARAMETER stop "{TemplateLoader.replace_and_concat(template, template.suffix, "", "")}"\n'
        return content

    @staticmethod
    def _read_content_from_url(url):
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
        return content.decode('utf-8')

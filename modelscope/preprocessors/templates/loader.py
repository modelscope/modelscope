import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple

import requests

from modelscope import AutoTokenizer, get_logger, snapshot_download, AutoConfig
from . import TemplateType
from .base import Template, get_template

logger = get_logger()


@dataclass
class TemplateInfo:

    template: str = None
    template_regex: str = None
    modelfile_prefix: str = None


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


# Order matters
template_info = [
    # llama
    ## "llama3"
    TemplateInfo(
        template=TemplateType.llama3,
        template_regex=
        f'.*{cases("llama3.2", "llama-3.2")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama3.2',
    ),
    TemplateInfo(
        template=TemplateType.llama3,
        template_regex=
        f'.*{cases("llama3.1", "llama-3.1")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama3.1',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("llama3", "llama-3")}.*{no_multi_modal()}.*{chat_suffix}.*{cases("gradient")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama3-gradient',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("llama3", "llama-3")}.*{no_multi_modal()}.*{cases("groq")}.*{cases("tool-use", "tool_use")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama3-groq-tool-use',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("llama3", "llama-3")}.*{no_multi_modal()}.*{cases("chatqa")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama3-chatqa',
    ),
    TemplateInfo(
        template_regex=f'.*{cases("llava-llama-3")}.*',
        modelfile_prefix='https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llava-llama3'),
    TemplateInfo(
        template_regex=f'.*{cases("dolphin")}.*{cases("llama3")}.*',
        modelfile_prefix='https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/dolphin-llama3'),
    TemplateInfo(
        template=TemplateType.llama3,
        template_regex=
        f'.*{cases("llama3", "llama-3")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama3',
    ),

    ## "llama"
    TemplateInfo(
        template_regex=
        f'.*{cases("llama2", "llama-2")}{no_multi_modal()}.*{cases("chinese")}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama2-chinese',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("codellama")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/codellama',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("tinyllama")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/tinyllama',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("llama-pro", "llama_pro")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama-pro',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("llama")}.*{cases("guard")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama-guard3',
    ),
    TemplateInfo(
        template=TemplateType.llama,
        template_regex=
        f'.*{cases("llama")}{no_multi_modal()}.*{chat_suffix}.*',
                modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llama2',
    ),

    # qwen
    TemplateInfo(
        template=TemplateType.qwen,
        template_regex=f'.*{cases("qwen2.5")}.*{cases("coder")}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/qwen2.5-coder',
    ),
    TemplateInfo(
        template=TemplateType.qwen,
        template_regex=f'.*{cases("qwen2.5")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/qwen2.5',
    ),
    TemplateInfo(
        template_regex=f'.*{cases("qwen2-math")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/qwen2-math',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("codeqwen1.5", "codeqwen-1.5")}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/codeqwen',
    ),
    TemplateInfo(
        template=TemplateType.qwen,
        template_regex=f'.*{cases("qwen2", "qwen1.5")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/qwen2',
    ),
    TemplateInfo(
        template=TemplateType.qwen,
        template_regex=f'.*{cases("qwen")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/qwen',
    ),

    # gemma
    TemplateInfo(
        template_regex=
        f'.*{cases("codegemma")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/codegemma',
    ),
    TemplateInfo(
        template=TemplateType.gemma,
        template_regex=
        f'{no("pali")}.*{cases("gemma2", "gemma-2")}\\b.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/gemma2',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("shieldgemma")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/shieldgemma',
    ),
    TemplateInfo(
        template=TemplateType.gemma,
        template_regex=
        f'{no("pali")}.*{cases("gemma")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/gemma',
    ),

    # "dolphin"
    TemplateInfo(
        template_regex=
        f'.*{cases("dolphin")}.*{cases("-mixtral")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/dolphin-mixtral',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("dolphin")}.*{cases("mistral")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/dolphin-mistral',
    ),

    # "phi"
    TemplateInfo(
        template_regex=
        f'.*{cases("llava-phi3", "llava-phi-3")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llava-phi3',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("phi3.5", "phi-3.5")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/phi3.5',
    ),
    TemplateInfo(
        template=TemplateType.phi3,
        template_regex=
        f'.*{cases("phi3", "phi-3")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/phi3',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("phi")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/phi',
    ),

    # "mistral"
    TemplateInfo(
        template_regex=
        f'.*{cases("yarn")}.*{cases("mistral")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/yarn-mistral',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("mistral")}.*{cases("large")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistral-large',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("mistral")}.*{cases("small")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistral-small',
    ),
    TemplateInfo(
        template=TemplateType.mistral_nemo,
        template_regex=f'.*{cases("Mistral-Nemo")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistral-nemo',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("mistral")}.*{cases("openorca")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistral-openorca',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("mistrallite")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistrallite',
    ),
    ## other mistral: set Type.llama
    TemplateInfo(
        template=TemplateType.llama,
        template_regex=
        f'.*{cases("mistral")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mistral',
    ),

    # "mixtral"
    TemplateInfo(
        template_regex=
        f'.*{cases("nous-hermes2", "nous-hermes-2")}.*{cases("mixtral")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nous-hermes2-mixtral',
    ),
    TemplateInfo(
        template=TemplateType.llama,
        template_regex=
        f'.*{cases("mixtral")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mixtral',
    ),

    # codestral
    TemplateInfo(
        template=TemplateType.llama,
        template_regex=
        f'.*{cases("codestral")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/codestral',
    ),

    # nous-hermes2
    TemplateInfo(
        template_regex=
        f'.*{cases("nous-hermes2", "nous-hermes-2")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nous-hermes2',
    ),
        TemplateInfo(
        template_regex=f'.*{cases("nous-hermes")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nous-hermes'),

    # "deepseek"
    TemplateInfo(
        template=TemplateType.deepseek2_5,
        template_regex=
        f'.*{cases("deepseek")}.*{cases("v2.5")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek-v2.5',
    ),
    TemplateInfo(
        template=TemplateType.deepseek_coder,
        template_regex=
        f'.*{cases("deepseek")}.*{cases("coder")}.*{cases("v2")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek-coder-v2',
    ),
    TemplateInfo(
        template=TemplateType.deepseek_coder,
        template_regex=
        f'.*{cases("deepseek")}{no("v2", "v2.5")}.*{cases("coder")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek-coder',
    ),
    TemplateInfo(
        template=TemplateType.deepseek2,
        template_regex=
        f'.*{cases("deepseek")}.*{cases("v2")}{no("v2.5")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek-v2',
    ),
    TemplateInfo(
        template=TemplateType.deepseek,
        template_regex=
        f'.*{cases("deepseek")}{no("v2", "v2.5", "coder")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek-llm',
    ),

    # "yi"
    TemplateInfo(
        template=TemplateType.yi_coder,
        template_regex=f'.*{cases("yi")}.*{cases("coder")}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/yi-coder',
    ),
    TemplateInfo(
        template=TemplateType.chatml,
        template_regex=
        f'.*{cases("yi")}{no_multi_modal()}{no("coder")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/yi',
    ),

    # "llava"
    TemplateInfo(
        template_regex=
        f'.*{cases("bakllava")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/bakllava',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("llava")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/llava',
    ),

    # "nemotron"
    TemplateInfo(
        template_regex=
        f'.*{cases("nemotron-mini")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nemotron-mini',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("nemotron")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nemotron',
    ),

    # "minicpm"
    TemplateInfo(
        template_regex=f'.*{cases("minicpm-v")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/minicpm-v'
    ),
    TemplateInfo(
        template=TemplateType.chatml,
        template_regex=f'.*{cases("minicpm")}{no("-v")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/yi'
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
        template_regex=f'.*{cases("glm4", "glm-4")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/glm4',
    ),

    # baichuan
    TemplateInfo(
        template=TemplateType.baichuan,
        template_regex=
        f'.*{cases("baichuan")}{no_multi_modal()}.*{chat_suffix}.*'),

    # "command-r"
    TemplateInfo(
        template_regex=
        f'.*{cases("command-r-plus")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/command-r-plus',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("command-r")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/command-r',
    ),

    # codegeex
    TemplateInfo(
        template=TemplateType.codegeex4,
        template_regex=f'.*{cases("codegeex4")}{no_multi_modal()}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/codegeex4',
    ),

    # wizard
    TemplateInfo(
        template_regex=
        f'.*{cases("wizard-vicuna")}.*{cases("uncensored")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/wizard-vicuna-uncensored',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("wizardlm2", "wizardlm-2")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/wizardlm2',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("wizardcoder")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/wizardcoder',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("wizard-math", "wizardmath")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/wizard-math',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("wizardlm")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/wizardlm',
    ),

    # vicuna
    TemplateInfo(
        template_regex=
        f'.*{cases("vicuna")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/vicuna',
    ),

    # "stable"
    TemplateInfo(
        template_regex=
        f'.*{cases("stable-code")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/stable-code',
    ),
    TemplateInfo(
        template_regex=
        f'.*{cases("stablelm")}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/stablelm2',
    ),

    # idefics3
    TemplateInfo(
        template=TemplateType.idefics3,
        template_regex=f'.*{cases("idefics3")}{no_multi_modal()}.*'),

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
        f'.*{cases("internlm2")}{no_multi_modal()}.*{chat_suffix}.*',
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/internlm2',
    ),

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
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/deepseek_v2',
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

    # telechat
    TemplateInfo(
        template=TemplateType.telechat,
        template_regex=f'.*{cases("TeleChat")}{no("v2")}.*'),

    # telechat_v2
    TemplateInfo(
        template=TemplateType.telechat_v2,
        template_regex=f'.*{cases("TeleChat")}.*{cases("v2")}.*'),

    TemplateInfo(
        template_regex=f'.*{cases("nomic-embed-text")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nomic-embed-text'),
    TemplateInfo(
        template_regex=f'.*{cases("mxbai-embed-large")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mxbai-embed-large'),
    TemplateInfo(
        template_regex=f'.*{cases("starcoder2")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/starcoder2'),
    TemplateInfo(
        template_regex=f'.*{cases("orca-mini", "orca_mini")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/orca-mini'),
    TemplateInfo(
        template_regex=f'.*{cases("zephyr")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/zephyr'),
    TemplateInfo(
        template_regex=f'.*{cases("snowflake-arctic-embed")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/snowflake-arctic-embed'),
    TemplateInfo(
        template_regex=f'.*{cases("starcoder")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/starcoder'),
    TemplateInfo(
        template_regex=f'.*{cases("granite")}.*{cases("code")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/granite-code'),
    TemplateInfo(
        template_regex=f'.*{cases("all-minilm")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/all-minilm'),
    TemplateInfo(
        template_regex=f'.*{cases("openchat")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/openchat'),
    TemplateInfo(
        template_regex=f'.*{cases("aya")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/aya'),
    TemplateInfo(
        template_regex=f'.*{cases("openhermes")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/openhermes'),
    TemplateInfo(
        template_regex=f'.*{cases("reflection")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/reflection'),
    TemplateInfo(
        template_regex=f'.*{cases("neural-chat")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/neural-chat'),
    TemplateInfo(
        template_regex=f'.*{cases("moondream")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/moondream'),
    TemplateInfo(
        template_regex=f'.*{cases("xwin")}.*{cases("lm")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/xwinlm'),
    TemplateInfo(
        template_regex=f'.*{cases("smollm")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/smollm'),
    TemplateInfo(
        template_regex=f'.*{cases("sqlcoder")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/sqlcoder'),
    TemplateInfo(
        template_regex=f'.*{cases("starling-lm")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/starling-lm'),
    TemplateInfo(
        template_regex=f'.*{cases("falcon")}.*{cases("-2")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/falcon2'),
    TemplateInfo(
        template_regex=f'.*{cases("falcon")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/falcon'),
    TemplateInfo(
        template_regex=f'.*{cases("solar-pro")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/solar-pro'),
    TemplateInfo(
        template_regex=f'.*{cases("solar")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/solar'),
    TemplateInfo(
        template_regex=f'.*{cases("orca2", "orca-2", "orca_2")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/orca2'),
    TemplateInfo(
        template_regex=f'.*{cases("hermes3", "hermes-3", "hermes_3")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/hermes3'),
    TemplateInfo(
        template_regex=f'.*{cases("meditron")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/meditron'),
    TemplateInfo(
        template_regex=f'.*{cases("nexusraven")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nexusraven'),
    TemplateInfo(
        template_regex=f'.*{cases("magicoder")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/magicoder'),
    TemplateInfo(
        template_regex=f'.*{cases("bge-m3")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/bge-m3'),
    TemplateInfo(
        template_regex=f'.*{cases("notux")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/notux'),
    TemplateInfo(
        template_regex=f'.*{cases("open")}.*{cases("orca")}.*{cases("platypus2")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/open-orca-platypus2'),
    TemplateInfo(
        template_regex=f'.*{cases("notus")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/notus'),
    TemplateInfo(
        template_regex=f'.*{cases("mathstral")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/mathstral'),
    TemplateInfo(
        template_regex=f'.*{cases("dbrx")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/dbrx'),
    TemplateInfo(
        template_regex=f'.*{cases("nuextract")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/nuextract'),
    TemplateInfo(
        template_regex=f'.*{cases("reader-lm")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/reader-lm'),
    TemplateInfo(
        template_regex=f'.*{cases("alfred")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/alfred'),
    TemplateInfo(
        template_regex=f'.*{cases("bge-large")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/bge-large'),
    TemplateInfo(
        template_regex=f'.*{cases("paraphrase-multilingual")}.*', 
        modelfile_prefix=
        'https://modelscope.oss-cn-beijing.aliyuncs.com/llm_template/ollama/paraphrase-multilingual'),

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
        config = kwargs.get('config')
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
                            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
                        except Exception:
                            pass
                    return TemplateLoader.load_by_template_name(
                        _info.template, tokenizer=tokenizer, config=config, **kwargs)

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
        template = get_template(template_name, tokenizer=kwargs.pop('tokenizer', None), **kwargs)
        template.config = kwargs.get('config')
        return template

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
    def _format_return(template_lines: str, params: Dict, split: bool, license: Optional[str] = None) -> Union[str, Dict]:
        if split:
            if params:
                params = json.dumps(params)
            return {'params': params, 'template': template_lines, 'license': license}

        content = ''
        content += 'FROM {gguf_file}\n\n'
        if params:
            for key, values in params.items():
                if isinstance(values, list):
                    for value in values:
                        content += f'PARAMETER {key} {json.dumps(value)}\n'
                else:
                    content += f'PARAMETER {key} {json.dumps(values)}\n'
            content += '\n'
        if template_lines:
            content += ('TEMPLATE """' + template_lines + '"""\n')
        return content

    @staticmethod
    def to_ollama(model_id: str = None,
                  template_name: str = None,
                  gguf_file: str = None,
                  gguf_meta: Dict[str, Any] = None,
                  split: bool = False,
                  debug: bool = False,
                  **kwargs) -> Union[str, Dict, Tuple[Dict, TemplateInfo], Tuple[str, TemplateInfo], None]:
        """Export to ollama ModelFile

        Args:
            model_id: The model-id to use
            template_name: An extra template name to use
            gguf_file: An extra gguf_file path to use in the `FROM` field
            gguf_meta: An gguf extra meta info
            split: bool. Return str modelfile content, or dict of params and template
            debug: bool. Whether or not to return the matched TemplateInfo
        Returns:
            The ModelFile content, or dictionary of params and template, returns `None` if no template found
        """

        if not model_id and not template_name and not gguf_meta:
            raise ValueError(
                f'Please make sure you model_id: {model_id} '
                f'and template_name: {template_name} is supported.')
        logger.info('Exporting to ollama:')
        names = []
        if gguf_meta:
            gguf_header_name = gguf_meta.get("general.name", None)
            names.append(gguf_header_name)
        if model_id:
            names.append(model_id)
        for name in names:
            for _info in template_info:
                if re.fullmatch(_info.template_regex, name):
                    if _info.modelfile_prefix and not kwargs.get('ignore_oss_model_file', False):
                        template_str = TemplateLoader._read_content_from_url(
                            _info.modelfile_prefix + '.template')
                        if not template_str:
                            logger.info(f'{name} has no template file.')
                        params = TemplateLoader._read_content_from_url(_info.modelfile_prefix + '.params')
                        if params:
                            params = json.loads(params)
                        else:
                            logger.info(f'{name} has no params file.')
                        license = TemplateLoader._read_content_from_url(
                            _info.modelfile_prefix + '.license')
                        if not template_str:
                            logger.info(f'{name} has no license file.')
                        format_out = TemplateLoader._format_return(template_str, params, split, license)
                        if debug:
                            return format_out, _info
                        return format_out
        if template_name:
            template = TemplateLoader.load_by_template_name(
                template_name, **kwargs)
        else:
            template = TemplateLoader.load_by_model_id(
                model_id, **kwargs)

        if not template:
            return None
            
        # template
        template_lines = ''
        _prefix = TemplateLoader.replace_and_concat(template, template.prefix, "", "")
        if _prefix:
            template_lines += (
                f'{{{{ if .System }}}}'
                f'{TemplateLoader.replace_and_concat(template, template.system_prefix or [], "{{SYSTEM}}", "{{ .System }}")}'
                f'{{{{ else }}}}{_prefix}'
                f'{{{{ end }}}}')
        else:
            template_lines += (
                f'{{{{ if .System }}}}'
                f'{TemplateLoader.replace_and_concat(template, template.system_prefix or [], "{{SYSTEM}}", "{{ .System }}")}'
                f'{{{{ end }}}}')
        template_lines += (
            f'{{{{ if .Prompt }}}}'
            f'{TemplateLoader.replace_and_concat(template, template.prompt, "{{QUERY}}", "{{ .Prompt }}")}'
            f'{{{{ end }}}}')
        template_lines += '{{ .Response }}'
        template_lines += TemplateLoader.replace_and_concat(template, template.suffix,
                                                     '', '')
        # stop tokens
        all_eos_tokens = {TemplateLoader.replace_and_concat(template, template.suffix, "", "")}
        if getattr(template, 'tokenizer', None):
            eos_token = TemplateLoader.replace_and_concat(template, [["eos_token_id"]], "", "")
            all_eos_tokens.add(eos_token)
            if getattr(template, 'config', None) and getattr(template.config, 'eos_token_id'):
                eos_token_id = template.config.eos_token_id
                eos_token = TemplateLoader.replace_and_concat(template, [[eos_token_id]], "", "")
                all_eos_tokens.add(eos_token)

        stop_tokens = list()
        for eos_token in all_eos_tokens:
            stop_tokens.append(eos_token)
        params = {'stop': stop_tokens}

        return TemplateLoader._format_return(template_lines, params, split)

    @staticmethod
    def _read_content_from_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            return None
        content = response.content
        return content.decode('utf-8')

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""several datasets with preset arguments"""
import os
import random
from multiprocessing import Process, Queue

import json
import tqdm
from torch.utils import data

from .datasets import csv_dataset, json_dataset
from .lazy_loader import LazyLoader

NUM_PROCESSES = 40


class webtext(json_dataset):
    """
    dataset for webtext with arguments configured for convenience

    command line usage: `--train-data webtext`
    """
    PATH = 'data/webtext/data.json'
    assert_str = 'make sure to set PATH for webtext data_utils/corpora.py'

    def __init__(self, **kwargs):
        assert os.path.exists(webtext.PATH), \
            webtext.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(webtext, self).__init__(webtext.PATH, **kwargs)


class KeyDataset(data.Dataset):

    def __init__(self, text_loader, mask_loader, **kwargs):
        self.texts = text_loader
        self.masks = mask_loader
        self.is_lazy = False
        if isinstance(self.texts, LazyLoader) and isinstance(
                self.masks, LazyLoader):
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        return self.text_lens[idx]

    def __getitem__(self, index):
        text = self.texts[index]
        mask_length = self.masks[index]
        mask = []
        for i, length in enumerate(mask_length):
            if i % 2 == 0:
                mask += [0] * length
            else:
                mask += [1] * length
        assert len(text) == len(mask)
        return {'tokens': text, 'loss_masks': mask}

    def __len__(self):
        return len(self.texts)


class PromptDataset(data.Dataset):

    def __init__(self,
                 prompt_loader,
                 text_loader,
                 tokenizer=None,
                 to_tokenize=False,
                 **kwargs):
        self.prompts = prompt_loader
        self.texts = text_loader
        self.tokenizer = tokenizer
        self.to_tokenize = to_tokenize
        if isinstance(self.prompts, LazyLoader) and isinstance(
                self.texts, LazyLoader):
            self.prompt_lens = self.prompts.lens
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        return self.prompt_lens[idx] + self.text_lens[idx]

    def __getitem__(self, index):
        prompt = self.prompts[index]
        text = self.texts[index]
        if self.to_tokenize:
            prompt = self.tokenizer.EncodeAsIds(prompt).tokenization
            text = self.tokenizer.EncodeAsIds(text).tokenization
        return {
            'tokens': prompt + text,
            'loss_masks': [0] * len(prompt) + [1] * len(text)
        }

    def __len__(self):
        return len(self.prompts)


class DataReader:
    PATH = None
    assert_str = None

    @staticmethod
    def tokenize_worker(input, output, reader, tokenizer, tokenize):
        raise NotImplementedError

    def __init__(self, writers, tokenizer=None, tokenize=False, **kwargs):
        assert os.path.exists(self.PATH), self.assert_str
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.writers = writers
        if os.path.isdir(self.PATH):
            paths = [
                entry.path for entry in os.scandir(self.PATH)
                if not entry.is_dir() and not entry.name.endswith('bz2')
            ]
        else:
            paths = [self.PATH]
        task_queue, done_queue = Queue(), Queue()
        processes = []
        for i in range(NUM_PROCESSES):
            process = Process(
                target=self.tokenize_worker,
                args=(task_queue, done_queue, type(self), tokenizer, tokenize))
            process.start()
            processes.append(process)
        for path in paths:
            with open(path) as file:
                for row in tqdm.tqdm(file):
                    task_queue.put(row)
        for i in range(len(processes)):
            task_queue.put('STOP')
        count = len(processes)
        progress_bar = tqdm.tqdm()
        while True:
            data = done_queue.get()
            if data == 'COMPLETE':
                count -= 1
                if count == 0:
                    break
            else:
                self.write_result(data, self.writers)
                progress_bar.update()
        progress_bar.close()

    @staticmethod
    def write_result(data, writers):
        raise NotImplementedError

    @staticmethod
    def get_token_count(contents):
        return sum(map(len, contents))

    @staticmethod
    def process_sample(text, tokenizer, tokenize):
        if isinstance(text, str) and tokenize:
            text = tokenizer.EncodeAsIds(text).tokenization if text else []
        return text

    @staticmethod
    def trim_field(content, max_length):
        if len(content) > max_length:
            content = content[:max_length]
            content += '......'
        return content

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        raise NotImplementedError


class PromptReader(DataReader):

    @staticmethod
    def tokenize_worker(input, output, reader, tokenizer, tokenize):
        for row in iter(input.get, 'STOP'):
            data = json.loads(row)
            prompts, texts = reader.process_line(data, tokenizer, tokenize)
            for prompt, text in zip(prompts, texts):
                output.put((prompt, text))
        output.put('COMPLETE')

    @staticmethod
    def write_result(data, writers):
        prompt, text = data
        writers['prompt'].write(prompt)
        writers['text'].write(text)


class KeyReader(DataReader):
    PATH = '/root/data/wikipedia/wiki-key.txt'
    assert_str = 'make sure to set PATH for wikipedia data_utils/corpora.py'

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        keys, contents = data['key'], data['content']
        assert len(keys) == len(contents)
        for i in range(1, len(keys)):
            keys[i] = ' ' + keys[i]
        contents = [' ' + content for content in contents]
        keys = [tokenizer.EncodeAsIds(key).tokenization for key in keys]
        contents = [
            tokenizer.EncodeAsIds(content).tokenization for content in contents
        ]
        summary = sum(keys, [])
        summary_prefix = cls.process_sample('Summary: ', tokenizer, tokenize)
        summary_mask = [len(summary_prefix), len(summary)]
        summary = summary_prefix + summary
        text, text_mask = [], []
        for key, content in zip(keys, contents):
            text += key
            text += content
            text_mask.append(len(key))
            text_mask.append(len(content))
        return (summary, summary_mask), (text, text_mask)

    @staticmethod
    def tokenize_worker(input, output, reader, tokenizer, tokenize):
        for row in iter(input.get, 'STOP'):
            data = json.loads(row)
            summary, content = reader.process_line(data, tokenizer, tokenize)
            output.put((summary, content))
        output.put('COMPLETE')

    @staticmethod
    def write_result(data, writers):
        summary, content = data
        writers['text'].write(summary[0])
        writers['mask'].write(summary[1])
        writers['text'].write(content[0])
        writers['mask'].write(content[1])


class zhihu(PromptReader):
    PATH = '/root/data/zhihu/zhihu'
    # PATH = "data/zhihu/data.json"
    assert_str = 'make sure to set PATH for zhihu data_utils/corpora.py'
    qtitle_prefix = '问题：'
    qcontent_prefix = '问题描述：'
    user_prefix = '回答用户：'
    answer_prefix = ' 回答：'

    # qtitle_prefix = []
    # qcontent_prefix = []
    # user_prefix = []
    # answer_prefix = []

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        prompts, texts = [], []
        ans_length = len(data.get('ans-content', ''))
        ans_up = data.get('ans-up-num', '')
        ans_up = int(ans_up) if ans_up else 0
        if ans_length > 100 or ans_up > 1000:
            qtitle = data['q_title']
            qcontent = data['q-content']
            if qcontent is None:
                qcontent = ''
            qcontent = cls.trim_field(qcontent, max_length=100)
            user = data.get('user-signature', '')
            prompt = cls.qtitle_prefix + qtitle + cls.qcontent_prefix + qcontent + cls.user_prefix + user + cls.answer_prefix  # noqa
            text = data['ans-content']
            prompt, text = cls.process_sample(prompt, tokenizer,
                                              tokenize), cls.process_sample(
                                                  text, tokenizer, tokenize)
            prompts.append(prompt)
            texts.append(text)
        # prompt = data["q_title"] + data["q-content"] + data["user-signature"]
        # text = data["ans-content"]
        # prompts.append(prompt)
        # texts.append(text)
        return prompts, texts


class zhidao(PromptReader):
    PATH = '/root/data/zhidao/zhidao'
    assert_str = 'make sure to set PATH for zhidao data_utils/corpora.py'
    qtitle_prefix = '问题：'
    qcontent_prefix = '问题描述：'
    answer_prefix = '回答：'

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        if 'title' not in data:
            return [], []
        prompts, texts = [], []
        qtitle = data['title']
        qcontent = data.get('content', '')
        qcontent = cls.trim_field(qcontent, max_length=100)
        prompt = cls.qtitle_prefix + qtitle + cls.qcontent_prefix + qcontent + cls.answer_prefix
        prompt = cls.process_sample(prompt, tokenizer, tokenize)
        if 'best_answer' in data:
            text = data['best_answer']['content']
            if len(text) > 10:
                text = cls.process_sample(text, tokenizer, tokenize)
                prompts.append(prompt)
                texts.append(text)
        for answer in data.get('other_answers', []):
            text = answer['content']
            if len(text) > 100:
                text = cls.process_sample(text, tokenizer, tokenize)
                prompts.append(prompt)
                texts.append(text)
        return prompts, texts


class baike(PromptReader):
    PATH = '/root/data/baike/baike'
    assert_str = 'make sure to set PATH for baike data_utils/corpora.py'

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        prompts, texts = [], []
        text = data.get('title', '') + data.get('abstract', '') + data.get(
            'content', '')
        if text:
            p, t = cls.process_sample('', tokenizer,
                                      tokenize), cls.process_sample(
                                          text, tokenizer, tokenize)
            prompts.append(p)
            texts.append(t)
        return prompts, texts


class wikipedia(PromptReader):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    # PATH = '/dataset/data/wiki.txt'
    PATH = '/root/data/wikipedia/wiki.txt'
    assert_str = 'make sure to set PATH for wikipedia data_utils/corpora.py'

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        text = data['text']
        prompt, text = cls.process_sample('', tokenizer,
                                          tokenize), cls.process_sample(
                                              text, tokenizer, tokenize)
        return [prompt], [text]


NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'wikipedia-key': KeyReader,
    'webtext': webtext,
    'zhihu': zhihu,
    'zhidao': zhidao,
    'baike': baike
}

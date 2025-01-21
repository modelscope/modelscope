# Copyright (c) 2022 Zhipu.AI

import glob
import os

import json
import nltk


class NLTKSegmenter:

    def __init(self):
        download_nltk()

    @staticmethod
    def segment_string(article):
        return nltk.tokenize.sent_tokenize(article)


def download_nltk():
    nltk.download('punkt')
    wiki_path = 'data/extracted'
    output_path = 'formatted/wiki-key.txt'
    segmenter = NLTKSegmenter()
    with open(output_path, 'w') as output:
        for dirname in glob.glob(
                os.path.join(wiki_path, '*'), recursive=False):
            for filename in glob.glob(
                    os.path.join(dirname, 'wiki_*'), recursive=True):
                print(filename)
                article_lines = []
                article_open = False
                with open(
                        filename, mode='r', newline='\n',
                        encoding='utf-8') as file:
                    for line in file:
                        line = line.rstrip()
                        if '<doc id=' in line:
                            article_open = True
                        elif '</doc>' in line:
                            key_sentences, contents = [], []
                            key, content = None, []
                            for sentences in article_lines[1:]:
                                if len(sentences) > 1:
                                    if key:
                                        if len(content) > 0 or len(
                                                contents) == 0:
                                            key_sentences.append(key)
                                            contents.append(content)
                                        else:
                                            contents[-1].append(key)
                                        key, content = None, []
                                    key_sentences.append(sentences[0])
                                    contents.append(sentences[1:])
                                elif len(sentences) > 0:
                                    if key:
                                        content.append(sentences[0])
                                    else:
                                        key = sentences[0]
                            if key:
                                if len(content) > 0 or len(contents) == 0:
                                    key_sentences.append(key)
                                    contents.append(content)
                                else:
                                    contents[-1].append(key)
                            contents = [
                                ' '.join(content) for content in contents
                            ]
                            article = {
                                'key': key_sentences,
                                'content': contents
                            }
                            output.write(json.dumps(article))
                            output.write('\n')
                            article_open = False
                            article_lines = []
                        else:
                            if article_open and line:
                                sentences = segmenter.segment_string(line)
                                article_lines.append(sentences)

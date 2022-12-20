# Copyright (c) Alibaba, Inc. and its affiliates.

import codecs
import re
import unicodedata

WordPattern = r'((?P<Word>\w+)(\(\w+\))?)'
BreakPattern = r'(?P<Break>(\*?#(?P<BreakLevel>[0-4])))'
MarkPattern = r'(?P<Mark>[、，。！？：“”《》·])'
POSPattern = r'(?P<POS>(\*?\|(?P<POSClass>[1-9])))'
PhraseTonePattern = r'(?P<PhraseTone>(\*?%([L|H])))'

NgBreakPattern = r'^ng(?P<break>\d)'

RegexWord = re.compile(WordPattern + r'\s*')
RegexBreak = re.compile(BreakPattern + r'\s*')
RegexID = re.compile(r'^(?P<ID>[a-zA-Z\-_0-9\.]+)\s*')
RegexSentence = re.compile(r'({}|{}|{}|{}|{})\s*'.format(
    WordPattern, BreakPattern, MarkPattern, POSPattern, PhraseTonePattern))
RegexForeignLang = re.compile(r'[A-Z@]')
RegexSpace = re.compile(r'^\s*')
RegexNeutralTone = re.compile(r'[1-5]5')


def do_character_normalization(line):
    return unicodedata.normalize('NFKC', line)


def do_prosody_text_normalization(line):
    tokens = line.split('\t')
    text = tokens[1]
    # Remove punctuations
    text = text.replace(u'。', ' ')
    text = text.replace(u'、', ' ')
    text = text.replace(u'“', ' ')
    text = text.replace(u'”', ' ')
    text = text.replace(u'‘', ' ')
    text = text.replace(u'’', ' ')
    text = text.replace(u'|', ' ')
    text = text.replace(u'《', ' ')
    text = text.replace(u'》', ' ')
    text = text.replace(u'【', ' ')
    text = text.replace(u'】', ' ')
    text = text.replace(u'—', ' ')
    text = text.replace(u'―', ' ')
    text = text.replace('.', ' ')
    text = text.replace('!', ' ')
    text = text.replace('?', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('~', ' ')
    text = text.replace(':', ' ')
    text = text.replace(';', ' ')
    text = text.replace('+', ' ')
    text = text.replace(',', ' ')
    #    text = text.replace('·', ' ')
    text = text.replace('"', ' ')
    text = text.replace(
        '-',
        '')  # don't replace by space because compond word like two-year-old
    text = text.replace(
        "'", '')  # don't replace by space because English word like that's

    # Replace break
    text = text.replace('/', '#2')
    text = text.replace('%', '#3')
    # Remove useless spaces surround #2 #3 #4
    text = re.sub(r'(#\d)[ ]+', r'\1', text)
    text = re.sub(r'[ ]+(#\d)', r'\1', text)
    # Replace space by #1
    text = re.sub('[ ]+', '#1', text)

    # Remove break at the end of the text
    text = re.sub(r'#\d$', '', text)

    # Add #1 between target language and foreign language
    text = re.sub(r"([a-zA-Z])([^a-zA-Z\d\#\s\'\%\/\-])", r'\1#1\2', text)
    text = re.sub(r"([^a-zA-Z\d\#\s\'\%\/\-])([a-zA-Z])", r'\1#1\2', text)

    return tokens[0] + '\t' + text


def is_fp_line(line):
    fp_category_list = ['FP', 'I', 'N', 'Q']
    elements = line.strip().split(' ')
    res = True
    for ele in elements:
        if ele not in fp_category_list:
            res = False
            break
    return res


def format_prosody(src_prosody):
    formatted_lines = []
    with codecs.open(src_prosody, 'r', 'utf-8') as f:
        lines = f.readlines()
        fp_enable = is_fp_line(lines[1])

        for i in range(0, len(lines)):
            line = do_character_normalization(lines[i])
            if fp_enable:
                if i % 5 == 1 or i % 5 == 2 or i % 5 == 3:
                    continue
            if len(line.strip().split('\t')) == 2:
                line = do_prosody_text_normalization(line)
            formatted_lines.append(line)
    return formatted_lines

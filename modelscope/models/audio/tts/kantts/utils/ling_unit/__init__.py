# Copyright (c) Alibaba, Inc. and its affiliates.

import ttsfrd


def text_to_mit_symbols(texts, resources_dir, speaker):
    fe = ttsfrd.TtsFrontendEngine()
    fe.initialize(resources_dir)
    fe.set_lang_type('Zh-CN')

    symbols_lst = []
    for idx, text in enumerate(texts):
        text = text.strip()
        res = fe.gen_tacotron_symbols(text)
        res = res.replace('F7', speaker)
        sentences = res.split('\n')
        for sentence in sentences:
            arr = sentence.split('\t')
            # skip the empty line
            if len(arr) != 2:
                continue
            sub_index, symbols = sentence.split('\t')
            symbol_str = '{}_{}\t{}\n'.format(idx, sub_index, symbols)
            symbols_lst.append(symbol_str)

    return symbols_lst

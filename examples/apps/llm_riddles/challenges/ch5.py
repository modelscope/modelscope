def check_word_in_sentence(words, sentence):
    return [word in sentence for word in words]


challenge5 = {
    'name':
    '第五章 登堂入室',
    'problems': [
        {
            'title':
            '第1题 盛夏少年',
            'description':
            '模型的回答应该包含“盛夏”、“蝉鸣”、“少年”、“橘子味汽水”这几个词，同时输入的问题不能包含其中任一个词。',
            'validator':
            lambda response, input: all(
                check_word_in_sentence(['盛夏', '蝉鸣', '少年', '橘子味汽水'], response))
            and not any(
                check_word_in_sentence(['盛夏', '蝉鸣', '少年', '橘子味汽水'], input))
        },
        {
            'title':
            '第2题 蝉鸣日出',
            'description':
            '模型的回答应该包含“盛夏”、“蝉鸣”、“少年”、“橘子味汽水”、“日出”这几个词，同时输入的问题不能包含其中任一个字。',
            'validator':
            lambda response, input: all(
                check_word_in_sentence(
                    ['盛夏', '蝉鸣', '少年', '橘子味汽水', '日出'], response)) and not any(
                        check_word_in_sentence([
                            '盛', '夏', '蝉', '鸣', '少', '年', '橘', '子', '味', '汽',
                            '水', '日', '出'
                        ], input))
        },
    ]
}

import re
import unicodedata


def validate_palindrome_inverse(origin_input, response, generate_response):
    """
    验证器函数，检查对于给定的问题，正着问和倒着问的模型的回答本身不回文且也是逆序的关系。

    :param origin_input: 原始问题
    :param response: 模型对原始问题的回答
    :param model_fn: 能够返回模型回答的函数
    :return: True 如果回答本身不回文且也是逆序关系，否则 False
    """

    # 检查原始输入不是回文
    if origin_input == origin_input[::-1]:
        return False

    # 将原始问题倒序并获取模型回答
    reversed_question = origin_input[::-1]
    reversed_response = generate_response(reversed_question).strip()

    # 检查原始回答不是回文
    if response == response[::-1]:
        return False

    # 比较正着问和倒着问的回答是否是逆序关系
    return response == reversed_response[::-1]


challenge3 = {
    'name':
    '第三章 巅峰挑战',
    'problems': [
        {
            'title':
            '第1题 自说自话',
            'description':
            '请做出一个至少10个字的提问，使回答和提问完全一样。',
            'validator':
            lambda response, input: input == response and len(input) >= 10
        },
        {
            'title':
            '第2题 话自说自',
            'description':
            '请做出一个至少10个字的提问，使回答是提问的逆序。',
            'validator':
            lambda response, input:
            (input[::-1] == response and len(input) >= 10)
        },
        {
            'title':
            '第3题 就是敢说',
            'description':
            '请在不提及1,3这两个字符的情况下让模型回答1+1=3，不要有多余的字符。',
            'validator':
            lambda response, input: response.strip() == '1+1=3' and all(
                c not in input for c in ['1', '3'])
        },
        {
            'title':
            '第4题 回文协变',
            'description':
            '请输入一个本身不是回文串的问题，使正着问和倒着问的时候，模型的回答本身不回文且也是逆序的关系。',
            'validator':
            lambda response, input, generate_response:
            validate_palindrome_inverse(input, response, generate_response)
        },
        {
            'title':
            '第5题 答非所问',
            'description':
            '请输入一个以“为什么”开头，以“？”结尾的不包含数字的问题，使回答中只包含数字。',
            'validator':
            lambda response, input: '为什么' == input[:3] and '？' == input[-1]
            and  # noqa: W504
            len(re.findall(r'\d+', input)) == 0 and response.isdigit()
        },
        {
            'title':
            '第6题 与狗交谈',
            'description':
            '请在不提及“狗”、“汪”、“dog”的情况下，让模型的回答是一字不差的“汪~”',
            'validator':
            lambda response, input:
            (all(char not in input
                 for char in ['狗', '汪', 'dog']) and response.strip() == '汪~')
        },
        {
            'title':
            '第7题 魔搭 （🐶）',
            'description':
            '请提一个不包含任何中文字符、换行或空格的问题，使模型正好回答"魔搭"。',
            'validator':
            lambda response, input: not any(
                'CJK' in unicodedata.name(char, '') or char in '\t\n '
                for char in input) and (response.strip() == '魔搭')
        },
    ]
}

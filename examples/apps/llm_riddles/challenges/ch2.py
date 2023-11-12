import re

import sympy


# 辅助函数 - 检查是否为质数
def is_prime(num):
    return sympy.isprime(num)


# 辅助函数 - 获取下一个质数
def next_prime(num):
    return sympy.nextprime(num)


# 辅助函数 - 检查是否为平方数
def is_square(n):
    return sympy.sqrt(n).is_integer


# 辅助函数 - 获取平方根
def get_square_root(n):
    return int(sympy.sqrt(n))


challenge2 = {
    'name':
    '第二章 数字游戏',
    'problems': [
        {
            'title':
            '第1题 质数长度',
            'description':
            '你需要提出一个字数是质数的问题，使回答的长度刚好是它的下一个质数。',
            'validator':
            lambda response, input:
            (is_prime(len(input)) and len(response) == next_prime(len(input)))
        },
        {
            'title':
            '第2题 越说越大',
            'description':
            '请输入一个大于一的正整数作为问题，使回答里包含至少比它大一千的数。',
            'validator':
            lambda response, input:
            (input.isdigit() and int(input) > 1 and any(
                int(num) > int(input) + 1000
                for num in re.findall(r'\d+', response)))
        },
        {
            'title':
            '第3题 越说越小',
            'description':
            '请输入一个大于一的正整数作为问题，使回答里包含至少10个不同的至少比它小一千的自然数。',
            'validator':
            lambda response, input:
            (input.isdigit() and int(input) > 1 and (len({
                int(num)
                for num in re.findall(r'\d+', response)
                if int(num) < int(input) - 1000
            }) >= 10))
        },
        {
            'title':
            '第4题 警世名言',
            'description':
            '请在不提及1,4,5这三个字符的情况下让模型回答114514，不要有多余的字符。',
            'validator':
            lambda response, input: response.strip() == '114514' and all(
                c not in input for c in ['1', '4', '5'])
        },
        {
            'title':
            '第5题 开平方数',
            'description':
            '请输入一个大于一的平方数作为问题，使回答里包含它的平方根对应的数。',
            'validator':
            lambda response, input: ((
                input.isdigit() and int(input) > 1 and is_square(int(input))
                and  # noqa: W504
                str(get_square_root(int(input))) in response))
        },
        {
            'title':
            '第6题 得寸进狗',
            'description':
            '请提一个不包含“狗”这个字的10个字以内的问题，使回答中“狗”这个字出现的次数至少是问题字数的2倍。',
            'validator':
            lambda response, input: (
                '狗' not in input and len(input) <= 10 and  # noqa: W504
                response.count('狗') >= 2 * len(input))
        },
        {
            'title':
            '第7题 暗藏玄机',
            'description':
            '请输入一个大于一的正整数作为问题，使回答包含一个两位数，其十位上的数字比个位上的数字小5。',
            'validator':
            lambda response, input:
            (input.isdigit() and int(input) > 1 and (len({
                int(num)
                for num in re.findall(r'\d+', response)
                if len(num) == 2 and int(num[1]) - int(num[0]) == 5
            }) >= 1))
        },
        {
            'title':
            '第8题 点兵点将',
            'description':
            '请输入十个字以内的问题，问题中不能包含0到9的数字以及零到十的汉字，使模型的回答中包含数字0到9。',
            'validator':
            lambda response, input: (
                len(input) <= 10 and
                not re.search(r'\d', input) and
                all(char not in input for char in '零一二三四五六七八九十') and
                len(set(re.findall(r'\d', response))) == 10)
        },
    ]
}

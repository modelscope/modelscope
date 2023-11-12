import re


def validate_reciprocal_question(input, response, generate_response):
    """
    验证器函数，检查给定的问题A和回答B，是否能够通过以B作为新的提问得到原始问题A作为回答。

    :param response: 模型对问题A的回答B
    :param input: 原始问题A
    :param model_fn: 模型调用函数，接收问题并返回回答
    :return: True 如果以回答B作为新的提问能得到问题A作为回答，否则 False
    """

    # 确保回答B不等于原始问题A
    if response == input:
        return False

    # 以回答B作为新的提问
    new_response = generate_response(response)

    # 检查新的回答是否与原始问题A相等
    return new_response.strip() == input.strip()


challenge4 = {
    'name':
    '第四章 无人之境',
    'problems': [
        {
            'title':
            '第1题 循环往复',
            'description':
            '请构造一个问题A，它的回复是不同于A的B，然后以B提问能再次得到A.',
            'validator':
            lambda response, input, generate_response:
            validate_reciprocal_question(input, response, generate_response)
        },
        {
            'title':
            '第2题 惜字如金',
            'description':
            '请输入一个字的问题，使模型的回答在16个字以内。',
            'validator':
            lambda response, input: len(input) == 1 and len(response) <= 16
        },
        {
            'title':
            '第3题 自然之密',
            'description':
            '请输入一个大于一的正整数作为问题，使回答里包含和它刚好相差1的数。',
            'validator':
            lambda response, input:
            (input.isdigit() and int(input) > 1 and any(
                abs(int(num) - int(input)) == 1
                for num in re.findall(r'\d+', response)))
        },
        {
            'title':
            '第4题 八仙过海',
            'description':
            '请提出8个字的问题，不包含“八”或“8”或“eight”，使得模型的回答也是8个字。',
            'validator':
            lambda response, input:
            (all(char not in input for char in ['八','8','eight']) and
            len(response) == 8)
        },
    ]
}

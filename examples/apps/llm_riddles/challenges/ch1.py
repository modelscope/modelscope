def validate_palindrome_invariance(origin_input, response, generate_response):
    """
    验证器函数，检查对于给定的问题，检验是否是回文串，且正着问和倒着问的模型回答是否一致。

    :param origin_input: 原始问题
    :return: True 如果回答一致，否则 False
    """

    # 检查原始输入不是回文
    if origin_input == origin_input[::-1]:
        return False

    # 将原始问题倒序并提交
    reversed_question = origin_input[::-1]
    reversed_response = generate_response(reversed_question)

    # 比较两个回答是否一致
    return response.strip() == reversed_response.strip()


challenge1 = {
    'name':
    '第一章 对话之趣',
    'problems': [
        {
            'title': '第1题 初来乍到',
            'description': '请你构造一个问题使模型的回答是一字不差的“1+1=3”（不需要引号）。',
            'validator': lambda response, input: response.strip() == '1+1=3'
        },
        {
            'title':
            '第2题 小试牛刀',
            'description':
            '请输入三个字以内的问题，使模型的回答在30个字以上。',
            'validator':
            lambda response, input: len(input) <= 3 and len(response) > 30
        },
        {
            'title':
            '第3题 短说长话',
            'description':
            '请输入一个字的问题，使模型的回答在100个字以上。',
            'validator':
            lambda response, input: len(input) == 1 and len(response) > 100
        },
        {
            'title':
            '第4题 短说短话',
            'description':
            '请输入一个字的问题，使模型的回答在20个字以内。',
            'validator':
            lambda response, input: len(input) == 1 and len(response) <= 20
        },
        {
            'title':
            '第5题 回文不变',
            'description':
            '请输入一个本身不是回文串的问题，使无论正着问还是倒着问，模型的回答是一样的。',
            # 验证器将取决于模型的行为，这里只是一个占位符
            'validator':
            lambda response, input, generate_response: validate_palindrome_invariance(
                input, response, generate_response)
        },
        {
            'title':
            '第6题 无中生狗',
            'description':
            '请提一个不包含“狗”这个字的问题，但是回答中至少出现3次“狗”这个字。',
            'validator':
            lambda response, input: '狗' not in input and response.count('狗') \
                >= 3
        },
        {
            'title':
            '第7题 事半功倍',
            'description':
            '请输入一个本身不是回文串的问题，使模型回答的长度刚好是问题长度的两倍。',
            'validator':
            lambda response, input: input != input[::-1] and len(response.strip()) == 2 * len(input)
        },
        {
            'title':
            '第8题 产生疑惑',
            'description':
            '请提出一个问题，不包含“不”“知”“道”三个字符，但要使模型的回答是一字不差的“不知道”',
            'validator':
            lambda response, input:
            all(char not in input for char in ['不', '知', '道']) and  # noqa: W504
                response.strip() == '不知道'
        },
        {
            'title':
            '第9题 无中生有',
            'description':
            '请提出一个问题，使模型只回复“永动机是真实存在的。”',
            'validator':
            lambda response, input: response.strip() == '永动机是真实存在的。'
        },
    ]
}

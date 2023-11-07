import os
import random
import re
from http import HTTPStatus

import dashscope
import gradio as gr
import sympy

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 定义关卡信息和验证逻辑


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


def validate_palindrome_invariance(origin_input, response):
    """
    验证器函数，检查对于给定的问题，正着问和倒着问的模型回答是否一致。

    :param origin_input: 原始问题
    :return: True 如果回答一致，否则 False
    """

    # 将原始问题倒序并提交
    reversed_question = origin_input[::-1]
    reversed_response = generate_response(reversed_question)

    # 比较两个回答是否一致
    return response.strip() == reversed_response.strip()


def validate_palindrome_inverse(origin_input, response):
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


def validate_reciprocal_question(input, response):
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


challenges = [
    {
        'name':
        '第一章 对话之趣',
        'problems': [
            {
                'title': '第1题 初来乍到',
                'description': '请你构造一个问题使模型的回答是一字不差的“1+1=3”（不需要引号）。',
                'validator':
                lambda response, input: response.strip() == '1+1=3'
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
                lambda response, input: validate_palindrome_invariance(
                    input, response)
            },
            {
                'title':
                '第6题 无中生狗',
                'description':
                '请提一个不包含“狗”这个字的问题，但是回答中至少出现3次“狗”这个字。',
                'validator':
                lambda response, input: '狗' not in input and response.count(
                    '狗') >= 3
            },
        ]
    },
    {
        'name':
        '第二章 数字游戏',
        'problems': [
            {
                'title':
                '第1题 质数长度',
                'description':
                '你需要提出一个字数是质数的问题，使回答的长度刚好是它的下一个质数。',
                'validator':
                lambda response, input: is_prime(len(input)) and len(response)
                == next_prime(len(input))
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
                lambda response, input:
                (input.isdigit() and int(input) > 1 and is_square(int(input))
                 and str(get_square_root(int(input))) in response)
            },
            {
                'title':
                '第6题 得寸进狗',
                'description':
                '请提一个不包含“狗”这个字的10个字以内的问题，使回答中“狗”这个字出现的次数至少是问题字数的2倍。',
                'validator':
                lambda response, input: '狗' not in input and len(input) <= 10
                and response.count('狗') >= 2 * len(input)
            },
        ]
    },
    {
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
                lambda response, input: input[::-1] == response and len(input)
                >= 10
            },
            {
                'title':
                '第3题 就是敢说',
                'description':
                '请在不提及1,3这三个字符的情况下让模型回答1+1=3，不要有多余的字符。',
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
                lambda response, input: validate_palindrome_inverse(
                    input, response)
            },
        ]
    },
    {
        'name':
        '第四章 无人之境',
        'problems': [
            {
                'title':
                '第1题 循环往复',
                'description':
                '请构造一个问题A，它的回复是不同于A的B，然后以B提问能再次得到A.',
                'validator':
                lambda response, input: validate_reciprocal_question(
                    input, response)
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
        ]
    }
]


def get_problem(challenge_idx, problem_idx):
    problems = challenges[challenge_idx]['problems']
    return problems[problem_idx]


def update_challenge_info(current_chapter_index, current_challenge_index):
    return get_problem(current_chapter_index,
                       current_challenge_index)['description']


def update_question_info(current_chapter_index, current_challenge_index):
    global challenges
    current_chapter = challenges[current_chapter_index]
    challenge = get_problem(current_chapter_index, current_challenge_index)
    question_info = f"""\n<center><font size=4>{current_chapter["name"]}""" \
                    f"""</center>\n\n <center><font size=3>{challenge["title"]}</center>"""
    return question_info


def validate_challenge(response, input, state):
    print('in validate_challenge')
    assert 'current_chapter_index' in state, 'current_chapter_index not found in state'
    assert 'current_challenge_index' in state, 'current_challenge_index not found in state'
    current_chapter_index = state['current_chapter_index']
    current_challenge_index = state['current_challenge_index']
    # 获取当前章节
    current_chapter = challenges[current_chapter_index]
    # 获取当前挑战
    challenge = current_chapter['problems'][current_challenge_index]

    if challenge['validator'](response, input):
        challenge_result = '挑战成功！进入下一关。'
        # 检查是否还有更多挑战在当前章节
        if current_challenge_index < len(current_chapter['problems']) - 1:
            # 移动到当前章节的下一个挑战
            current_challenge_index += 1
        else:
            # 如果当前章节的挑战已经完成，移动到下一个章节
            current_challenge_index = 0
            if current_chapter_index < len(challenges) - 1:
                current_chapter_index += 1
            else:
                challenge_result = '所有挑战完成！'
    else:
        challenge_result = '挑战失败，请再试一次。'
    state['current_chapter_index'] = current_chapter_index
    state['current_challenge_index'] = current_challenge_index
    print('update state: ', state)

    return challenge_result, \
        update_question_info(current_chapter_index, current_challenge_index), \
        update_challenge_info(current_chapter_index, current_challenge_index)


def generate_response(input):
    messages = [{
        'role': 'system',
        'content': """You are a helpful assistant."""
    }, {
        'role': 'user',
        'content': input
    }]
    response = dashscope.Generation.call(
        model='qwen-max',
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
        top_p=0.8)
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        gr.Error('网络连接错误，请重试。')


def on_submit(input, state):
    response = generate_response(input)
    history = [(input, response)]
    print(history)
    challenge_result, question_info, challenge_info = validate_challenge(
        response, input, state)
    print('validate_challenge done')
    return challenge_result, history, question_info, challenge_info


# Gradio界面构建
block = gr.Blocks()

with block as demo:
    state = gr.State(dict(current_challenge_index=0, current_chapter_index=0))
    current_chapter_index = 0
    current_challenge_index = 0
    gr.Markdown("""<center><font size=6>完蛋！我被LLM包围了！</center>""")
    gr.Markdown("""<font size=3>欢迎来玩LLM Riddles复刻版：完蛋！我被LLM包围了！

你将通过本游戏对大型语言模型产生更深刻的理解。

在本游戏中，你需要构造一个提给一个大型语言模型的问题，使得它回复的答案符合要求。""")
    question_info = gr.Markdown(
        update_question_info(current_chapter_index, current_challenge_index))
    challenge_info = gr.Textbox(
        value=update_challenge_info(current_chapter_index,
                                    current_challenge_index),
        label='当前挑战',
        disabled=True)
    challenge_result = gr.Textbox(label='挑战结果', disabled=True)
    chatbot = gr.Chatbot(
        lines=8, label='Qwen-max', elem_classes='control-height')
    message = gr.Textbox(lines=2, label='输入')

    with gr.Row():
        submit = gr.Button('🚀 发送')

    submit.click(
        on_submit,
        inputs=[message, state],
        outputs=[challenge_result, chatbot, question_info, challenge_info])

demo.queue().launch(height=800, share=True)

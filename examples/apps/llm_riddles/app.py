import functools
import inspect
import os
import random
import re

import gradio as gr
from challenges.ch1 import challenge1
from challenges.ch2 import challenge2
from challenges.ch3 import challenge3
from challenges.ch4 import challenge4
from llm import create_model

model_cache = {}

# 定义关卡信息和验证逻辑
challenges = [
    challenge1,
    challenge2,
    challenge3,
    challenge4,
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


def validate_challenge(response, input, state, generate_response):
    print('in validate_challenge')
    assert 'current_chapter_index' in state, 'current_chapter_index not found in state'
    assert 'current_challenge_index' in state, 'current_challenge_index not found in state'
    current_chapter_index = state['current_chapter_index']
    current_challenge_index = state['current_challenge_index']
    # 获取当前章节
    current_chapter = challenges[current_chapter_index]
    # 获取当前挑战
    challenge = current_chapter['problems'][current_challenge_index]

    validate_fn = challenge['validator']
    params = inspect.signature(validate_fn).parameters
    if 'generate_response' in params:
        valid_result = validate_fn(response, input, generate_response)
    else:
        valid_result = validate_fn(response, input)

    if valid_result:
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


def generate_response(input, model_name):
    if model_name in model_cache:
        model = model_cache[model_name]
    else:
        model = create_model(model_name)
        model_cache[model_name] = model

    try:
        return model(input)
    except RuntimeError as e:
        # if exception happens, print error in log and return empty str
        print('error', e)
        return ''


def on_submit(input, state):
    model_name = 'qwen-plus'
    gen_fn = functools.partial(generate_response, model_name=model_name)
    response = gen_fn(input)
    history = [(input, response)]
    print(history)
    challenge_result, question_info, challenge_info = validate_challenge(
        response, input, state, gen_fn)
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

demo.queue(concurrency_count=10).launch(
    height=800, share=True, concurrency_limit=10)

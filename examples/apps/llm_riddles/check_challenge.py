from app import challenges, generate_response


def check_answer(chap_idx,
                 challenge_idx,
                 input='input',
                 model_name='qwen-max'):
    print('第{}章 第{}题'.format(chap_idx + 1, challenge_idx + 1))
    challenge = challenges[chap_idx]['problems'][challenge_idx]
    print(challenge['description'])
    val_fn = challenge['validator']
    response = generate_response(input, model_name)
    try:
        res = val_fn(response, input)
        print('input:\n', input)
        print('response:\n', response)
        print('validation result: ', res)
    except Exception:
        import traceback
        traceback.print_exc()
        print('failed')


if __name__ == '__main__':
    chap = 5
    ques = 1
    input = '请使用“盛 夏”、“蝉 鸣”、“少 年”、“橘 子味汽水”这几个词造句'
    check_answer(chap - 1, ques - 1, input)

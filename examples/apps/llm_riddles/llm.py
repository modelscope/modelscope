import os
import random
from http import HTTPStatus
from typing import Any, Dict, List, Union


class DashScope:
    """A class to interact with the Dashscope AI service for response generation.

    This class provides an interface to call a specific model from the Dashscope service
    to generate responses based on the input provided.

    Attributes:
        model (str): The name of the model to be used for generation.
    """

    def __init__(self, model_name: str = 'qwen-plus'):
        """Initializes the DashScope instance with a given model name.

        The constructor sets up the model name that will be used for response generation
        and initializes the Dashscope API key from environment variables.

        Args:
            model_name (str): The name of the model to be used. Defaults to 'qwen-plus'.
        """
        import dashscope  # Import dashscope module at runtime
        dashscope.api_key = os.getenv(
            'DASHSCOPE_API_KEY')  # Set the API key from environment variable
        self.model: str = model_name  # Assign the model name to an instance variable

    def __call__(self, input: Union[str, List[Dict[str, str]]],
                 **kwargs: Any) -> Union[str, None]:
        """Allows the DashScope instance to be called as a function.

        This method processes the input, sends it to the Dashscope service, and returns
        the generated response.

        Args:
            input (Union[str, List[Dict[str, str]]]): The input str to generate a
                response for. Can be a string or a list of messages.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[str, None]: The generated response from the model, or None if there is an error.

        Raises:
            RuntimeError: If there is an error in accessing the Dashscope service.
        """
        import dashscope  # Import dashscope module at runtime
        # Format the input into the required structure
        if isinstance(input, str):
            messages: List[Dict[str, str]] = [{
                'role':
                'system',
                'content':
                'You are a helpful assistant.'
            }, {
                'role': 'user',
                'content': input
            }]
        else:
            messages = input

        # Make a call to the Dashscope service with the processed input
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            seed=random.randint(1,
                                10000),  # Generate a random seed for each call
            result_format='message',  # Specify the format of the result
            top_p=kwargs.get('top_p',
                             0.8)  # Set the nucleus sampling parameter
        )
        # Check the response status code and return the generated response or raise an error
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        else:
            print('Error accessing dashscope, please try again.',
                  response.request_id, response.message)
            return ''


class ZhiPu:

    def __init__(self, model_name: str = 'chatglm_turbo'):
        """Initializes the ZhiPu instance with a given model name.

        The constructor sets up the model name that will be used for response generation
        and initializes the Dashscope API key from environment variables.

        Args:
            model_name (str): The name of the model to be used. Defaults to 'qwen-plus'.
        """
        import zhipuai  # Import dashscope module at runtime
        zhipuai.api_key = os.getenv(
            'ZHIPU_API_KEY')  # Set the API key from environment variable
        self.model: str = model_name  # Assign the model name to an instance variable

    def __call__(self, input: Union[str, List[Dict[str, str]]],
                 **kwargs: Any) -> Union[str, None]:
        """Allows the ZhiPu instance to be called as a function.

        {
            "code":200,
            "msg":"操作成功",
            "data":{
                "request_id":"8098024428488935671",
                "task_id":"8098024428488935671",
                "task_status":"SUCCESS",
                "choices":[
                    {
                        "role":"assistant",
                        "content":"\" 您好！作为人工智能助手，我很乐意为您提供帮助。请问您有什么问题或者需要解决的事情吗？您可以向我提问，我会尽力为您解答。\""
                    }
                ],
                "usage":{
                    "prompt_tokens":2,
                    "completion_tokens":32,
                    "total_tokens":34
                }
            },
            "success":true
            }
        """
        import zhipuai
        if isinstance(input, str):
            messages: List[Dict[str, str]] = [{
                'role': 'user',
                'content': input
            }]
        else:
            messages = input

        response = zhipuai.model_api.invoke(
            model=self.model,
            prompt=messages,
            top_p=0.7,
            temperature=0.9,
            return_type='text',
        )
        if response['code'] == 200:
            return response['data']['choices'][0]['content']
        else:
            print(f'{self.model} error: ', response)
            return ''


def create_model(model_name: str):
    """Factory function to create a DashScope model instance based on the model name.

    Args:
        model_name (str): The name of the model to create an instance of.

    Returns:
        DashScope: An instance of the DashScope class.

    Raises:
        ValueError: If the model name provided does not start with 'qwen'.
    """
    if model_name.startswith('qwen'):
        return DashScope(model_name)
    elif model_name.startswith('chatglm'):
        return ZhiPu(model_name)
    else:
        raise ValueError('Other model implementations need to be provided.')


if __name__ == '__main__':
    model = create_model('chatglm_turbo')
    print(model('输入'))

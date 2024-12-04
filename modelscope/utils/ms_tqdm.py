import inspect
import os
import threading
import time
from functools import wraps

from tqdm.auto import tqdm as old_tqdm


def timing_decorator(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取调用函数的文件信息
        frame = inspect.currentframe()
        try:
            # 获取调用函数的调用者的信息
            caller_frame = frame.f_back
            filename = os.path.basename(caller_frame.f_code.co_filename)
            line_number = caller_frame.f_lineno
        finally:
            del frame  # 明确删除以防止循环引用

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 打印丰富的调试信息
        print(
            f"Function '{func.__name__}' in {filename} - line {line_number}, took {elapsed_time:.4f} seconds."
        )

        return result

    return wrapper


class tqdm(old_tqdm):
    _lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        with self._lock:
            super().update(n)

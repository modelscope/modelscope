# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import wraps


def thread_executor(max_workers: int = min(32, os.cpu_count() + 4)):
    """
    A decorator to execute a function in a threaded manner using ThreadPoolExecutor.

    Args:
        max_workers (int): The maximum number of threads to use.

    Returns:
        function: A wrapped function that executes with threading and a progress bar.

    Examples:
        >>> from modelscope.utils.thread_utils import thread_executor
        >>> import time
        >>> @thread_executor(max_workers=8)
        ... def process_item(item):
        ...     # do something to single item
        ...     time.sleep(1)
        ...     print(f'Item: {item}')

        >>> items = ['a', 'b', 'c']
        >>> process_item(items)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(iterable, *args, **kwargs):
            # Create a tqdm progress bar with the total number of items to process
            with tqdm(total=len(iterable), desc=f'Processing {len(iterable)} items') as pbar:
                # Define a wrapper function to update the progress bar
                def progress_wrapper(item):
                    result = func(item, *args, **kwargs)
                    pbar.update(1)
                    return result

                # Use ThreadPoolExecutor to execute the wrapped function
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    executor.map(progress_wrapper, iterable)

        return wrapper

    return decorator
